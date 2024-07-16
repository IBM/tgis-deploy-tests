# test-gen

test-gen is a utility script to assist with generating common case files for multiple models.

## How it works

test-gen reads a tgis-tester config file, e.g. `config.product.yaml` and a common case definition file, e.g. `common.yaml`. It deploys TGIS worker pods (one at a time) for each base model config, submits Generate requests via port-forward for each case, and finally writes test case files.

As this is just a simple script, it does not include validation and error handling like tgis-tester.

### Usage

For each model, it runs the base config (i.e. index 0) only, writing to the base case definition file path.

```
Usage: test_gen.py [OPTIONS]

Options:
  --config-path TEXT        tgis-tester config file  [default:
                            ../config/product.yaml]
  --common-cases-path TEXT  common cases file  [default: common.yaml]
  --image-tag TEXT          TGIS image tag  [required]
  --models TEXT             specified models only
  --help                    Show this message and exit.
```

Run for all models:
```
python test_gen.py --image-tag <IMAGE_TAG>
```