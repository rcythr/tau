mod agent;
mod context;
mod harness;
mod llm;
mod orchestrator;
mod telemetry;
mod tools;

use agent::{Agent, AgentType};
use clap::Parser;
use llm::OpenAiClient;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "tau", about = "Local LLM agent")]
struct Cli {
    /// Single prompt (non-interactive mode)
    prompt: Option<String>,

    #[arg(long, env = "TAU_BASE_URL", default_value = "http://localhost:8080/v1")]
    base_url: String,

    #[arg(long, env = "TAU_API_KEY")]
    api_key: Option<String>,

    #[arg(long, env = "TAU_MODEL", default_value = "default")]
    model: String,

    #[arg(long, default_value_t = 50)]
    max_turns: usize,

    #[arg(long, default_value_t = 8192)]
    max_context: usize,

    #[arg(long)]
    system: Option<String>,

    /// Disable streaming (required until Phase 4)
    #[arg(long)]
    no_stream: bool,

    /// Verbose: print token counts etc.
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. init tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::WARN.into()),
        )
        .init();

    // 2. parse CLI
    let cli = Cli::parse();

    // 3. build OpenAiClient
    let client = OpenAiClient::new(&cli.base_url, cli.api_key.clone(), &cli.model);

    // 4. build AgentType
    let system_prompt = cli
        .system
        .clone()
        .unwrap_or_else(|| "You are a helpful assistant.".to_string());
    let agent_type = AgentType {
        system_prompt,
        model: cli.model.clone(),
        max_turns: cli.max_turns,
    };

    // 5. build Agent
    let mut agent = Agent::new(agent_type, Arc::new(client));

    // 6. run prompt
    let prompt = cli
        .prompt
        .as_deref()
        .unwrap_or("Hello!")
        .to_string();

    if cli.verbose {
        eprintln!("Running prompt: {}", prompt);
    }

    let result = agent.run(&prompt).await?;

    // 7. print result
    println!("{}", result);

    Ok(())
}
