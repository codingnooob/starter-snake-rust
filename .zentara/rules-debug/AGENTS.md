# Project Debug Rules (Non-Obvious Only)

## Server Port Confusion
- Rocket.toml sets port 8888, but README examples use 8000
- Actual port depends on PORT environment variable (main.rs:101-103)
- If server starts but client can't connect, check: PORT env var vs Rocket.toml vs client URL

## Logging Configuration
- Default log level is "info", set in code (main.rs:108) not config file
- RUST_LOG env var overrides this - use for debug output
- Game lifecycle logs: "INFO" (on /), "GAME START", "GAME OVER", "MOVE {turn}: {move}"
- If moves aren't being logged, check RUST_LOG isn't filtering them out

## Type Mismatch Errors
- Board.width (i32) vs Board.height (u32) causes unexpected type errors
- If bounds checks fail mysteriously, verify you're not comparing width/height directly
- Cast to common type before arithmetic: `width as u32` or `height as i32`

## Random Behavior Debugging
- RNG is not seeded (src/logic.rs:94) - moves are truly random
- Cannot reproduce specific game sequences without modifying RNG usage
- For debugging specific scenarios, replace rand::rng() with seeded RNG

## Body Index Panics
- Code assumes body.len() >= 2 (accesses body[1] for neck)
- Snake can have length 1 in edge cases (just spawned or about to die)
- If panic on body[1], add length check before neck access