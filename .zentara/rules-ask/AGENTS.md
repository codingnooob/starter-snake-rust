# Project Documentation Rules (Non-Obvious Only)

## Port Configuration Confusion
- Documentation mismatch: Rocket.toml=8888, README examples=8000
- Actual behavior: PORT env var is translated to ROCKET_PORT in code (main.rs:101-103)
- This is Battlesnake deployment convention, not documented in Rocket guides
- Always clarify which port is actually being used based on environment

## Data Structure Quirks
- Board has type inconsistency: width=i32, height=u32 (src/main.rs:27-28)
- This appears unintentional but affects type inference throughout code
- When explaining bounds checking, must address this casting requirement

## Battlesnake Body Semantics
- body vector ordering is head-to-tail: [0]=head, [1]=neck, [2+]=body
- This is implicit in Battlesnake API but not documented in struct definitions
- Code relies on body[0] and body[1] access without length checks (potential panic)

## Testing Limitations
- No test suite included in starter project
- Random number generation is unseeded (src/logic.rs:94)
- Cannot reproduce specific game scenarios without code modification
- When explaining testing strategy, must address RNG seeding requirement

## Environment Variables
- RUST_LOG controls logging, but code sets default "info" level (main.rs:108)
- Not standard Rust practice to set log level in code vs config
- PORT→ROCKET_PORT translation is custom, not Rocket framework behavior