fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_client(true)
        .build_server(false)
        .out_dir("src")
        .compile(
            &["common/protobuf/grpc_service.proto"],
            &["common/protobuf"],
        )
        .unwrap_or_else(|e| panic!("protobuf compilation failed: {}", e));

    Ok(())
}
