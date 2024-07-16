use std::{sync::Arc, time::Duration};

use k8s_openapi::api::core::v1::Secret;
use object_store::ObjectStore;
use owo_colors::OwoColorize;
use tabled::{
    col,
    settings::{
        object::{Columns, Rows},
        themes::ColumnNames,
        Format, Modify, Padding, Style, Width,
    },
    Table,
};

use crate::{config::Quantize, evaluate::EvaluationStrategy, worker::Job, Config, Error, Run};

/// Parses model name from cache directory name.
pub fn parse_model_name(model_dir: impl AsRef<str>) -> Option<String> {
    let model_dir = model_dir.as_ref();
    model_dir
        .strip_prefix("models--")
        .map(|v| v.replace("--", "/"))
}

/// Builds a unique worker name.
pub fn worker_name(job: &Job) -> String {
    let mut name = format!(
        "t{}-{}-{}-{}",
        &job.run_id.as_simple().to_string()[0..4],
        job.model_name.short_name(),
        job.model_config.deployment_framework.short_name(),
        job.model_config.dtype.short_name(),
    );
    if job.model_config.merge_onnx_graphs.is_some() {
        name.push_str("-m")
    }
    if job.model_config.is_sharded() {
        name.push_str("-shard")
    }
    if job.model_config.is_cpu_only() {
        name.push_str("-cpu")
    }
    if job.model_config.pt2_compile {
        name.push_str("-pt2c")
    }
    if job.model_config.flash_attention {
        name.push_str("-flash")
    }
    if let Some(quantize) = &job.model_config.quantize {
        match quantize {
            Quantize::Gptq => name.push_str("-gptq"),
        }
    }
    name
}

/// Gets a secret.
pub async fn get_secret(kube: &kube::Client, secret_name: &str) -> Result<Secret, Error> {
    let secrets_api: kube::Api<Secret> = kube::Api::default_namespaced(kube.clone());
    secrets_api.get(secret_name).await.map_err(|e| e.into())
}

/// Gets a secret and extracts object store credentials.
pub async fn get_object_store_secret(
    kube: &kube::Client,
    secret_name: &str,
) -> Result<(String, String), Error> {
    let secret = get_secret(kube, secret_name).await?;
    let access_key = secret
        .data
        .as_ref()
        .unwrap()
        .get("access-key")
        .map(|v| String::from_utf8(v.0.clone()).unwrap())
        .unwrap();
    let secret_key = secret
        .data
        .as_ref()
        .unwrap()
        .get("secret-key")
        .map(|v| String::from_utf8(v.0.clone()).unwrap())
        .unwrap();
    Ok((access_key, secret_key))
}

/// Lists cached models.
pub async fn list_models(
    object_store: Arc<dyn ObjectStore>,
    prefix: impl AsRef<str>,
) -> Result<Vec<String>, Error> {
    let prefix = object_store::path::Path::parse(prefix).unwrap();
    let models = object_store
        .list_with_delimiter(Some(&prefix))
        .await?
        .common_prefixes
        .into_iter()
        .map(|p| {
            p.as_ref()
                .strip_prefix("transformers_cache/")
                .unwrap()
                .to_string()
        })
        .filter_map(parse_model_name)
        .collect::<Vec<_>>();
    Ok(models)
}

/// Validates that requested models exist in the cache.
pub async fn validate_models(
    config: &Config,
    object_store: Arc<dyn ObjectStore>,
) -> Result<(), Error> {
    let prefix = config
        .worker
        .transformers_cache_path
        .strip_prefix("/integration_tests")
        .unwrap();
    let cached_models = list_models(object_store, prefix).await?;
    for test in config.tests.iter() {
        if !cached_models.contains(&test.model_name) {
            return Err(Error::Validation(format!(
                "model `{}` not found in cache",
                &test.model_name
            )));
        }
    }
    Ok(())
}

pub fn run_plan_table(run: &Run) -> String {
    let rows = run
        .plan
        .iter()
        .map(|job| {
            [
                job.model_name.pretty_name(),
                job.model_config.name(),
                job.cases.len().to_string(),
            ]
        })
        .collect::<Vec<_>>();
    let run_details_table = Table::from_iter(vec![
        vec!["run id".to_string(), "workers".to_string()],
        vec![run.id.to_string(), rows.len().to_string()],
    ])
    .with(Style::modern())
    .with(ColumnNames::default())
    .to_string();
    let validate_models = if run.config.runner.validate_models {
        "✅".to_string()
    } else {
        "❌".to_string()
    };
    let generate_stream = if run.config.runner.generate_stream {
        "✅".to_string()
    } else {
        "❌".to_string()
    };
    let evaluation_strategy: Box<dyn EvaluationStrategy> =
        run.config.runner.evaluation_strategy.clone().into();
    let run_options_table = Table::from_iter([
        ["validate models".to_string(), validate_models],
        ["generate stream".to_string(), generate_stream],
        [
            "evaluation strategy".to_string(),
            evaluation_strategy.name().to_string(),
        ],
    ])
    .with(Style::modern().remove_horizontal())
    .with(ColumnNames::new(["run options", ""]))
    .to_string();
    let run_plan_table = Table::from_iter(rows)
        .with(Style::modern().remove_horizontal())
        .with(ColumnNames::new(["model", "config", "cases"]))
        .to_string();
    col![
        "Run Plan",
        run_details_table,
        run_options_table,
        run_plan_table
    ]
    .with(Style::modern().remove_horizontal())
    .with(Padding::new(1, 1, 0, 1))
    .to_string()
}

pub fn run_completed_table(run: &Run) -> String {
    let run_details_table = Table::from_iter(vec![
        vec![
            "run id".to_string(),
            "workers".to_string(),
            "duration".to_string(),
        ],
        vec![
            run.id.to_string(),
            run.workers().len().to_string(),
            duration_to_string(run.duration().unwrap()),
        ],
    ])
    .with(Style::modern())
    .with(ColumnNames::default())
    .to_string();
    col![
        "Run Completed",
        run_details_table,
        test_results_table(run),
        worker_state_table(run)
    ]
    .with(Style::modern().remove_horizontal())
    .with(Padding::new(1, 1, 0, 1))
    .to_string()
}

pub fn run_failed_table(run: &Run) -> String {
    let run_details_table = Table::from_iter(vec![
        vec![
            "run id".to_string(),
            "workers".to_string(),
            "duration".to_string(),
        ],
        vec![
            run.id.to_string(),
            run.workers().len().to_string(),
            run.duration()
                .map(duration_to_string)
                .unwrap_or("-".to_string()),
        ],
    ])
    .with(Style::modern())
    .with(ColumnNames::default())
    .to_string();
    col!["❌ Run Failed", run_details_table, worker_state_table(run)]
        .with(Style::modern().remove_horizontal())
        .with(Padding::new(1, 1, 0, 1))
        .to_string()
}

pub fn worker_state_table(run: &Run) -> String {
    let mut rows = run
        .workers()
        .iter()
        .map(|worker| {
            let mut model_name = worker.job().model_name.pretty_name();
            if model_name.len() > 19 {
                model_name.truncate(19);
                model_name.push_str("..");
            }
            let queued_duration = worker
                .create_start()
                .map(|create_start| create_start - run.start);
            vec![
                worker.name().to_string(),
                model_name,
                worker.model_config().name(),
                queued_duration
                    .map(duration_to_string)
                    .unwrap_or("-".to_string()),
                worker
                    .scheduled_duration()
                    .map(duration_to_string)
                    .unwrap_or("-".to_string()),
                worker
                    .ready_duration()
                    .map(duration_to_string)
                    .unwrap_or("-".to_string()),
                worker.state().to_string(),
            ]
        })
        .collect::<Vec<_>>();
    rows.sort_by(|a, b| b[6].cmp(&a[6]));
    let worker_state_table = Table::from_iter(rows)
        .with(Style::modern().remove_horizontal())
        .with(ColumnNames::new([
            "name".to_string(),
            "model".to_string(),
            "config".to_string(),
            "queued".to_string(),
            "scheduled".to_string(),
            "ready".to_string(),
            "state".to_string(),
        ]))
        .with(Modify::new(Columns::new(1..3)).with(Format::content(|s| s.dimmed().to_string())))
        .with(Modify::new(Rows::first()).with(Width::increase(8)))
        .to_string();
    col!["workers", worker_state_table]
        .with(Style::modern().remove_horizontal())
        .with(ColumnNames::default())
        .to_string()
}

pub fn test_results_table(run: &Run) -> String {
    let passed_ascii = r#"
█▀█ ▄▀█ █▀ █▀ █▀▀ █▀▄
█▀▀ █▀█ ▄█ ▄█ ██▄ █▄▀ 
    "#
    .green()
    .to_string();
    let failed_ascii = r#"
█▀▀ ▄▀█ █ █░░ █▀▀ █▀▄
█▀░ █▀█ █ █▄▄ ██▄ █▄▀
    "#
    .red()
    .to_string();
    let (_passed, failed): (Vec<_>, Vec<_>) = run.results().iter().partition(|&r| r.passed());
    if failed.is_empty() {
        col![passed_ascii, "All tests passed successfully 🎉"]
            .with(
                Style::modern()
                    .remove_horizontal()
                    .remove_left()
                    .remove_right()
                    .remove_bottom()
                    .remove_top(),
            )
            .to_string()
    } else {
        let rows = failed
            .iter()
            .enumerate()
            .map(|(i, &result)| {
                [
                    i.to_string(),
                    result.model_name.pretty_name(),
                    result.model_config.name(),
                    result.name.clone(),
                ]
            })
            .collect::<Vec<_>>();
        let failed_tests_table = Table::from_iter(rows)
            .with(Style::modern().remove_horizontal())
            .with(ColumnNames::new(["", "model", "config", "case"]))
            .with(Modify::new(Columns::new(0..1)).with(Format::content(|s| s.dimmed().to_string())))
            .to_string();
        col![
            failed_ascii,
            format!("{} tests failed", failed.len()),
            failed_tests_table
        ]
        .with(
            Style::modern()
                .remove_horizontal()
                .remove_left()
                .remove_right()
                .remove_bottom()
                .remove_top(),
        )
        .to_string()
    }
}

pub fn truncate(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        None => s,
        Some((idx, _)) => &s[..idx],
    }
}

pub fn duration_to_string(duration: Duration) -> String {
    let secs_f32 = duration.as_secs_f32();
    if secs_f32 < 1.0 {
        format!("{}ms", duration.as_millis())
    } else if (1.0..60.0).contains(&secs_f32) {
        format!("{}s", duration.as_secs())
    } else {
        format!("{}m", duration.as_secs() / 60)
    }
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use crate::{config::DeploymentFramework, ModelConfig, ModelName};

    use super::*;

    #[test]
    fn test_worker_name() {
        let run_id = Uuid::parse_str("2bcac63a-d2c8-4477-8964-e373fc384086").unwrap();
        let model_name = ModelName::new("bigscience/bloom-560m");
        let model_config = ModelConfig {
            revision: None,
            deployment_framework: DeploymentFramework::HfTransformers,
            dtype: Default::default(),
            num_shard: 1,
            port: 5001,
            grpc_port: 8001,
            master_addr: "localhost".to_string(),
            master_port: 9001,
            cuda_process_memory_fraction: 1.0,
            max_batch_size: 12,
            max_concurrent_requests: 96,
            max_sequence_length: 2048,
            max_new_tokens: 1024,
            max_waiting_tokens: 12,
            max_batch_weight: None,
            max_prefill_weight: None,
            output_special_tokens: false,
            flash_attention: false,
            pt2_compile: false,
            quantize: None,
            resources: Default::default(),
            merge_onnx_graphs: None,
            estimate_memory: None,
            enable_lora: false,
            gpu_memory_utilization: 0.85,
        };
        let job = Job {
            run_id,
            model_name,
            model_config: Arc::new(model_config),
            cases: Vec::new(),
            ca_cert: None,
            results: Vec::new(),
        };
        let name = worker_name(&job);
        assert_eq!(name, "t2bca-bloom560m-t-f16");
    }

    #[test]
    fn test_parse_model_name() {
        let model_dir = "models--bigscience--bloom-560m";
        let model_name = parse_model_name(model_dir);
        assert_eq!(model_name.as_deref(), Some("bigscience/bloom-560m"));
    }
}
