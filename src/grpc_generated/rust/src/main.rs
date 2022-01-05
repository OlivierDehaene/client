mod inference;

use inference::grpc_inference_service_client::GrpcInferenceServiceClient;
use inference::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};
use inference::{
    ModelInferRequest, ModelMetadataRequest, ModelMetadataResponse, ServerLiveRequest,
    ServerReadyRequest,
};
use std::str::FromStr;
use structopt::StructOpt;
use tonic::transport::Uri;
use tonic::{transport::Channel, Status};

#[derive(Debug, StructOpt)]
#[structopt(name = "Rust inference client")]
struct Opt {
    #[structopt(default_value = "simple", help = "Name of model being served.", short)]
    model_name: String,
    #[structopt(help = "Version of model. [default: Latest Version]", short = "x")]
    model_version: Option<String>,
    #[structopt(
        default_value = "http://localhost:8001",
        help = "Inference Server URL.",
        short
    )]
    url: String,
}

#[tracing::instrument]
async fn triton_live(client: &mut GrpcInferenceServiceClient<Channel>) -> Result<bool, Status> {
    let request = tonic::Request::new(ServerLiveRequest {});

    let response = client.server_live(request).await?;
    Ok(response.into_inner().live)
}

#[tracing::instrument]
async fn triton_ready(client: &mut GrpcInferenceServiceClient<Channel>) -> Result<bool, Status> {
    let request = tonic::Request::new(ServerReadyRequest {});

    let response = client.server_ready(request).await?;
    Ok(response.into_inner().ready)
}

#[tracing::instrument]
async fn triton_model_metadata(
    client: &mut GrpcInferenceServiceClient<Channel>,
    model_name: &str,
    model_version: &str,
) -> Result<ModelMetadataResponse, Status> {
    let request = tonic::Request::new(ModelMetadataRequest {
        name: model_name.to_string(),
        version: model_version.to_string(),
    });

    let response = client.model_metadata(request).await?;
    Ok(response.into_inner())
}

#[tracing::instrument]
async fn triton_infer(
    client: &mut GrpcInferenceServiceClient<Channel>,
    input_0: i32,
    input_1: i32,
    model_name: &str,
    model_version: &str,
) -> Result<(i32, i32), Status> {
    let infer_inputs = vec![
        InferInputTensor {
            name: "INPUT0".to_string(),
            datatype: "INT32".to_string(),
            shape: vec![1, 16],
            parameters: Default::default(),
            contents: None,
        },
        InferInputTensor {
            name: "INPUT1".to_string(),
            datatype: "INT32".to_string(),
            shape: vec![1, 16],
            parameters: Default::default(),
            contents: None,
        },
    ];
    let infer_outputs = vec![
        InferRequestedOutputTensor {
            name: "OUTPUT0".to_string(),
            parameters: Default::default(),
        },
        InferRequestedOutputTensor {
            name: "OUTPUT1".to_string(),
            parameters: Default::default(),
        },
    ];

    let mut triton_request = ModelInferRequest {
        model_name: model_name.to_string(),
        model_version: model_version.to_string(),
        id: "".to_string(),
        parameters: Default::default(),
        inputs: infer_inputs,
        outputs: infer_outputs,
        raw_input_contents: vec![],
    };

    triton_request
        .raw_input_contents
        .push(Vec::from(input_0.to_le_bytes()));
    triton_request
        .raw_input_contents
        .push(Vec::from(input_1.to_le_bytes()));

    let tonic_triton_request = tonic::Request::new(triton_request);
    let response = client.model_infer(tonic_triton_request).await?;

    let response = response.into_inner();
    let output_0 = <i32>::from_le_bytes(
        response.raw_output_contents[0]
            .as_slice()
            .try_into()
            .unwrap(),
    );
    let output_1 = <i32>::from_le_bytes(
        response.raw_output_contents[1]
            .as_slice()
            .try_into()
            .unwrap(),
    );

    Ok((output_0, output_1))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let Opt {
        model_name,
        model_version,
        url,
    } = Opt::from_args();
    let model_version = model_version.unwrap_or_default();

    let uri = Uri::from_str(&url)?;
    let channel = Channel::builder(uri).connect().await?;
    let mut client = GrpcInferenceServiceClient::new(channel);

    let live = triton_live(&mut client).await?;
    tracing::info!("Triton Health - Live: {}", live);

    let ready = triton_ready(&mut client).await?;
    tracing::info!("Triton Health - Ready: {}", ready);

    let metadata = triton_model_metadata(&mut client, &model_name, &model_version).await?;
    tracing::info!("{:?}", metadata);

    let (output0, output1) = triton_infer(&mut client, 0, 0, &model_name, &model_version).await?;
    tracing::info!("{} {}", output0, output1);

    Ok(())
}
