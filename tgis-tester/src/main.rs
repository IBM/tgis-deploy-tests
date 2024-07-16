use std::{fs::File, path::PathBuf, sync::Arc};

use clap::Parser;
use kube::config::AuthInfo;
use object_store::{aws::AmazonS3Builder, ObjectStore};

use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use tgis_tester::{
    utils::{get_object_store_secret, validate_models},
    Config, Error, Runner,
};

#[derive(Parser, Debug, Clone)]
struct Args {
    /// Path to the config.yml with cluster, runner, worker, and test case configurations
    #[arg(short, long, env)]
    config_path: PathBuf,
    /// The image tag to test. The base image name without the tag may be specified in the configuration file.
    #[arg(short, long, env, required_unless_present = "image_name")]
    image_tag: Option<String>,
    /// A full override of the image name to test. This may include a tag, in which case the --image-tag arg is ignored.
    #[arg(short, long, env, required_unless_present = "image_tag")]
    image_name: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    let args = Args::parse();
    println!("args: {:?}", args);
    let config = Arc::new(load_config(args)?);

    let filter = EnvFilter::try_from_default_env().unwrap_or(
        EnvFilter::new("INFO")
            .add_directive("cached_path=warn".parse().unwrap())
            .add_directive("object_store=warn".parse().unwrap())
            .add_directive("kube_client=error".parse().unwrap()),
    );
    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer())
        .init();

    let kube = {
        let token = config
            .cluster
            .service_account_token
            .as_ref()
            .ok_or(Error::InvalidConfig(
                "TESTER_SERVICE_ACCOUNT_TOKEN must be set".into(),
            ))?;
        let kube_config = kube::Config {
            cluster_url: config.cluster.api_endpoint.parse().unwrap(),
            default_namespace: config.cluster.namespace.clone(),
            root_cert: None,
            connect_timeout: None,
            read_timeout: None,
            write_timeout: None,
            accept_invalid_certs: true,
            auth_info: AuthInfo {
                username: Some(config.cluster.service_account_name.clone()),
                token: Some(token.clone()),
                ..Default::default()
            },
            proxy_url: None,
            tls_server_name: None,
        };
        kube::Client::try_from(kube_config).unwrap()
    };

    let object_store = {
        let (access_key, secret_key) =
            get_object_store_secret(&kube, &config.runner.object_store_secret).await?;
        let client = AmazonS3Builder::new()
            .with_endpoint(&config.runner.object_store_endpoint)
            .with_region(&config.runner.object_store_region)
            .with_bucket_name(&config.runner.object_store_bucket)
            .with_access_key_id(access_key)
            .with_secret_access_key(secret_key)
            .build()
            .unwrap();
        Arc::new(client) as Arc<dyn ObjectStore>
    };

    config.validate()?;
    if config.runner.validate_models {
        validate_models(&config, object_store.clone()).await?;
    }

    let mut runner = Runner::new(config, kube, object_store);
    let run = runner.run().await?;
    if !run.results.is_empty() {
        if run.results.iter().any(|r| !r.passed()) {
            let n_failed = run.results.iter().filter(|&r| !r.passed()).count();
            Err(Error::TestFailed(format!("{n_failed} tests failed")))
        } else {
            Ok(())
        }
    } else {
        Err(Error::TestFailed("no results".into()))
    }
}

fn load_config(args: Args) -> Result<Config, Error> {
    let file = File::open(&args.config_path).map_err(|e| Error::InvalidConfig(e.to_string()))?;
    let mut config = Config::load(file)?;

    // Update the image name if it was given
    if let Some(image_name) = args.image_name {
        config.worker.image_uri = image_name
    }

    // If the image name does not already contain a tag, append the given one
    if !config.worker.image_uri.contains(':') {
        if let Some(image_tag) = args.image_tag {
            config.worker.image_uri = format!("{}:{}", config.worker.image_uri, image_tag)
        }
    }

    let base_path = args.config_path.parent().unwrap().parent().unwrap();
    if let Some(ca_cert_path) = config.runner.ca_cert_path {
        let mut path = base_path.to_path_buf();
        path.push(ca_cert_path);
        config.runner.ca_cert_path = Some(path);
    }
    config.tests = config
        .tests
        .into_iter()
        .map(|mut spec| {
            let cases = spec
                .cases
                .iter()
                .map(|p| {
                    let mut path = base_path.to_path_buf();
                    path.push(p);
                    path
                })
                .collect::<Vec<_>>();
            spec.cases = cases;
            spec
        })
        .collect();
    Ok(config)
}
