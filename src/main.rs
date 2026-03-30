mod agent;
mod context;
mod harness;
mod llm;
mod orchestrator;
mod telemetry;
mod tools;

use agent::{Agent, AgentType};
use clap::Parser;
use context::{ContextPolicy, OutputCompressor};
use harness::{AllowList, CompositePolicy, DenyList, NoSandbox, ToolHarness};
use llm::OpenAiClient;
use std::sync::Arc;
use tools::ToolRegistry;

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

    /// Context trimming policy: "append-only", "pin-prefix", or "fresh"
    #[arg(long, default_value = "pin-prefix")]
    context_policy: String,

    #[arg(long)]
    system: Option<String>,

    /// Disable streaming (required until Phase 4)
    #[arg(long)]
    no_stream: bool,

    /// Verbose: print token counts etc.
    #[arg(short, long)]
    verbose: bool,

    /// Allow only these tools (repeatable). Omit to allow all.
    #[arg(long = "tool", value_name = "NAME")]
    tools_allow: Vec<String>,

    /// Deny these tools (repeatable).
    #[arg(long = "no-tool", value_name = "NAME")]
    tools_deny: Vec<String>,

    /// Sandbox mode: none, landlock, bwrap (default: none)
    #[arg(long, default_value = "none")]
    sandbox: String,
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

    let context_policy = match cli.context_policy.as_str() {
        "append-only" => ContextPolicy::AppendOnly,
        "fresh" => ContextPolicy::FreshContext,
        _ => ContextPolicy::PinPrefix { pinned: 4, recent: 20 },
    };

    // Build tool policy
    let harness = if !cli.tools_allow.is_empty() {
        // --tool flags: build AllowList with static lifetime via leak
        let names: Vec<&'static str> = cli
            .tools_allow
            .iter()
            .map(|s| Box::leak(s.clone().into_boxed_str()) as &'static str)
            .collect();
        ToolHarness::new(AllowList(names), NoSandbox)
    } else if !cli.tools_deny.is_empty() {
        // --no-tool flags: build DenyList
        let names: Vec<&'static str> = cli
            .tools_deny
            .iter()
            .map(|s| Box::leak(s.clone().into_boxed_str()) as &'static str)
            .collect();
        // Optionally compose with bash deny list for safety
        let deny = DenyList(names);
        ToolHarness::new(
            CompositePolicy(vec![Box::new(deny)]),
            NoSandbox,
        )
    } else {
        ToolHarness::permissive()
    };

    // Handle sandbox selection (landlock/bwrap are stubs for now)
    let harness = match cli.sandbox.as_str() {
        #[cfg(target_os = "linux")]
        "landlock" => {
            use harness::sandbox::LandlockSandbox;
            // We can't reuse harness above because of move, rebuild
            let _ = harness; // drop previous
            ToolHarness::new(
                harness::AllowAll,
                LandlockSandbox { allowed_rw_paths: vec![], allowed_ro_paths: vec![] },
            )
        }
        _ => harness,
    };

    let agent_type = AgentType {
        system_prompt,
        model: cli.model.clone(),
        max_turns: cli.max_turns,
        max_context: cli.max_context,
        context_policy,
        compressor: OutputCompressor::default(),
        tools: Arc::new(ToolRegistry::default()),
        harness: Arc::new(harness),
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
