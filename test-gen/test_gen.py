from typing import Optional, Any

import click
import kr8s
import grpc
import urllib3
from kr8s.objects import Pod
from google.protobuf import json_format
from progress.bar import IncrementalBar
from ruamel.yaml import YAML

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

yaml = YAML()
generation_pb2, generation_pb2_grpc = grpc.protos_and_services("generation.proto")

TGIS_IMAGE = "quay.io/wxpe/text-gen-server:main"
DEFAULT_CPU_REQUEST = 2
DEFAULT_CPU_LIMIT = 4
DEFAULT_MEMORY_REQUEST = 8
DEFAULT_MEMORY_LIMIT = 128
DEFAULT_CREATE_POD_TIMEOUT_SECS = 900
DEFAULT_DELETE_POD_TIMEOUT_SECS = 120
DEFAULT_DTYPE_STR = "float16"

def create_pod(pod: Pod, timeout_secs: Optional[int]=DEFAULT_CREATE_POD_TIMEOUT_SECS) -> Pod:
    """Creates pod and waits for it to be ready or fail."""
    pod.create()
    try:
        pod.wait(conditions=["condition=Failed", "condition=Ready"], timeout=timeout_secs)
    except TimeoutError:
        print("timeout waiting for pod to become ready")
        raise
    pod.refresh()
    return pod

def is_pod_failed(pod: Pod) -> bool:
    """Checks if pod has failed."""
    return pod.status.phase == "Failed"

def delete_pod(pod: Pod, timeout_secs: Optional[int]=DEFAULT_DELETE_POD_TIMEOUT_SECS) -> None:
    """Deletes pod and waits for it to be deleted."""
    pod.delete()
    try:
        pod.wait("delete", timeout=timeout_secs)
    except TimeoutError:
        print("timeout waiting for pod to delete")
        raise

def pod_manifest(
    image: str,
    model_name: str, 
    deployment_framework: str, 
    dtype_str: str = DEFAULT_DTYPE_STR, 
    flash_attention: bool = False,
    quantize: Optional[str] = None,
    output_special_tokens: Optional[bool] = False,
    gpu: Optional[str] = "nvidia.com/gpu",
    num_shard: Optional[int] = 1,
    n_gpus: Optional[int] = 1,
  ):
    """Builds pod manifest."""
    name = "test-gen"
    node_selector = {"nvidia.com/gpu.product": "NVIDIA-A100-SXM4-80GB"}
    envs = [
      ("HOME", "/home/user"),
      ("MODEL_NAME", model_name),
      ("DEPLOYMENT_FRAMEWORK", deployment_framework),
      ("DTYPE_STR", dtype_str),
      ("NUM_SHARD", f"{n_gpus}"),
      ("NUM_GPUS", f"{n_gpus}"),
      ("CUDA_VISIBLE_DEVICES", "0" if n_gpus == 1 else "0,1"),
      ("FLASH_ATTENTION", str(flash_attention).lower()),
      ("MAX_NEW_TOKENS", "1024"),
      ("MAX_SEQUENCE_LENGTH", "128"),
      ("MAX_BATCH_SIZE", "8"),
      ("MAX_CONCURRENT_REQUESTS", "96"),
      ("PORT", "5001"),
      ("GRPC_PORT", "8001"),
      ("VLLM_PORT", "8001"),
      ("TRANSFORMERS_CACHE", "/shared_model_storage/transformers_cache"),
      ("ESTIMATE_MEMORY", "off"),
    ]
    if quantize is not None:
        envs.append(("QUANTIZE", quantize))
    if output_special_tokens:
        envs.append(("OUTPUT_SPECIAL_TOKENS", "true"))
    envs = [{"name": name, "value": value} for (name, value) in envs]
    volumes = [
        {"name": "home", "emptyDir": {"medium": ""}}
    ]
    volumeMounts = [
        {"name": "home", "mountPath": "/home/user", "readOnly": False},
    ]
    resources = {
        "requests": {"memory": f"{DEFAULT_MEMORY_REQUEST}Gi", "cpu": str(DEFAULT_CPU_REQUEST), gpu: f"{n_gpus}"},
        "limits": {"memory": f"{DEFAULT_MEMORY_LIMIT}Gi", "cpu": str(DEFAULT_CPU_LIMIT), gpu: f"{n_gpus}"}
    }
    return {
      "apiVersion": "v1",
      "kind": "Pod",
      "metadata": {
          "labels": {
              "app": name,
            },
          "name": name,
      },
      "spec": {
          "nodeSelector": node_selector,
          "containers": [
              {
                  "name": "server",
                  "image": image,
                  "securityContext": {
                      "allowPrivilegeEscalation": False,
                      "privileged": False,
                      "runAsNonRoot": True,
                      "readOnlyRootFilesystem": True,
                      "seccompProfile": {
                          "type": "RuntimeDefault"
                      },
                      "capabilities": {
                          "drop": ["ALL"]
                      }
                  },
                  "ports": [{"name": "http", "containerPort": 5001, "name": "grpc", "containerPort": 8001}],
                  "env": envs,
                  "volumeMounts": volumeMounts,
                  "resources": resources,
                  "startupProbe": {
                      "httpGet": {"port": 5001, "path": "/health"},
                      "failureThreshold": 24,
                      "periodSeconds": 30,
                  },
                  "readinessProbe": {
                      "httpGet": {"port": 5001, "path": "/health"},
                      "periodSeconds": 30,
                      "timeoutSeconds": 5,
                  },
                  "livenessProbe": {
                      "httpGet": {"port": 5001, "path": "/health"},
                      "periodSeconds": 100,
                      "timeoutSeconds": 5,
                  },
              }
          ],
          "terminationGradePeriodSeconds": 120,
          "imagePullSecrets": [{"name": "artifactory-docker-token"}],
          "restartPolicy": "Never",
          "volumes": volumes,
      }
    }

def read_common_cases(path: str):
    """Read file with common cases."""
    with open(path, "r") as f:
        return yaml.load(f)

def read_config(path: str, models: Optional[list[str]]=None) -> list[dict[str, Any]]:
    """Reads model configs from tgis-tester config file."""
    with open(path, "r") as f:
        test_configs = yaml.load(f)["tests"]
    if models:
        test_configs = [t for t in test_configs if t["model"] in models]
    configs = []
    for t in test_configs:
        case_file = t["cases"][0]
        config = t["configs"][0]
        config["model_name"] = t["model"]
        resources = config.get("resources")
        if resources:
            config["n_gpus"] = resources.get("gpu", 1)
            del config["resources"]
        configs.append((case_file, config))
    return configs

def create_requests(cases: list[dict[str, Any]]):
    return [
        json_format.ParseDict(case["request"], 
            generation_pb2.BatchedGenerationRequest()) 
        for case in cases
    ]

def short_model_name(name: str) -> str:
    return name.split("/")[-1]

def create_case_files(cases: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> None:
    """Creates and writes case files with results."""
    for (key, responses) in outputs:
        output = []
        for (i, response) in enumerate(responses):
            case = cases[i].copy()
            response_dict = json_format.MessageToDict(response)
            for r in response_dict["responses"]:
                if "seed" in r:
                    r["seed"] = int(r["seed"])
            case["response"] = response_dict
            output.append(case)
        output_file = f"../{key}"
        with open(output_file, "w+") as f:
            yaml.dump(output, f)

@click.command()
@click.option("--config-path", default="../config/product.yaml", help="tgis-tester config file", show_default=True)
@click.option("--common-cases-path", default="common.yaml", help="common cases file", show_default=True)
@click.option("--image-tag", required=True, help="TGIS image tag")
@click.option("--models", callback=lambda _,__,x: x.split(',') if x else None, help="specified models only")
def main(config_path: str, common_cases_path: str, image_tag: str, models: Optional[list[str]]=None):
    k8s = kr8s.api()
    image = f"{TGIS_IMAGE}:{image_tag}"
    configs = read_config(config_path, models)
    cases = read_common_cases(common_cases_path)
    pods = [(config[0], Pod(pod_manifest(image, **config[1]))) for config in configs]
    requests = create_requests(cases)
    outputs = []
    with IncrementalBar("processing", max=len(pods)) as bar:
        for (name, pod) in pods:
            try:
                pod = create_pod(pod)
                if is_pod_failed(pod):
                    print(f"pod failed, deleting...")
                    delete_pod(pod)
                    raise
                with pod.portforward(remote_port=8001, local_port=8001):
                    with grpc.insecure_channel("localhost:8001") as channel:
                        stub = generation_pb2_grpc.GenerationServiceStub(channel)
                        responses = [stub.Generate(r) for r in requests]
                        outputs.append((name, responses))
            except Exception as e:
                raise
            delete_pod(pod)
            bar.next()

    print("writing case files")
    create_case_files(cases, outputs)
    print("completed")

if __name__ == "__main__":
    main()
