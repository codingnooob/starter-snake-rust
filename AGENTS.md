# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Build & Run
- `cargo run` - Starts server (actual port from PORT env var, falls back to Rocket.toml port 8888)
- `cargo test` - Run tests
- Local testing: `battlesnake play -W 11 -H 11 --name 'Rust Starter Project' --url http://localhost:8000 -g solo --browser`

## Port Configuration (Non-Obvious)
- Rocket.toml configures port 8888, but main.rs translates PORT env var to ROCKET_PORT (lines 101-103)
- README examples use port 8000 - actual port depends on which env var is set
- For deployment: Set PORT env var (not ROCKET_PORT) to match hosting provider expectations

## Type Inconsistencies
- Board.width is i32 but Board.height is u32 (src/main.rs:27-28) - likely unintentional
- Coord uses i32 for both x and y (line 48-49)
- When adding bounds checking, handle this type mismatch

## Battlesnake Body Structure
- you.body[0] is head position, you.body[1] is neck position (src/logic.rs:59-60)
- This indexing pattern is used for collision detection
- body vector is ordered from head to tail

## Random Number Generation
- Uses rand::rng() without explicit seeding (src/logic.rs:94)
- For deterministic testing, you'll need to inject a seeded RNG
- Current implementation: safe_moves.choose(&mut rand::rng())

## Logging
- Default log level is "info" (set in main.rs:108)
- Override with RUST_LOG env var
- Game events logged: "INFO", "GAME START", "GAME OVER", "MOVE {turn}: {chosen}"