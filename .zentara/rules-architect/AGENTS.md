# Project Architecture Rules (Non-Obvious Only)

## Server Configuration Pattern
- Custom PORT→ROCKET_PORT translation layer (main.rs:101-103)
- Required for Battlesnake cloud deployment compatibility
- Not standard Rocket pattern - hosting providers set PORT, Rocket expects ROCKET_PORT
- This translation must be maintained in main() before rocket::build()

## Type System Inconsistencies
- Board struct mixes i32 (width) and u32 (height) - likely unintentional
- Coord uses i32 for both x,y to match API, but creates casting issues
- When designing bounds checking or grid algorithms, must handle mixed signedness
- Consider: Should width be u32? Requires Battlesnake API compatibility check

## Stateless Game Logic
- All game state passed via function parameters (no global state)
- start(), get_move(), end() receive full GameState each call
- Snake "memory" between turns must be external (not in this codebase)
- Architecture enables horizontal scaling but limits strategic planning

## Body Structure Coupling
- Movement logic tightly coupled to body[0]=head, body[1]=neck assumption
- No validation of body.len() >= 2 before accessing body[1]
- Risk: Edge cases (spawning, dying) may violate assumption
- When adding collision detection, must handle variable body lengths

## Random Move Selection Architecture
- Uses global unseeded RNG (rand::rng()) in movement logic
- Makes game behavior non-deterministic and untestable
- For production: inject RNG dependency to enable seeded testing
- Current architecture prevents replay/debug of specific game scenarios

## Logging Strategy
- Hardcoded "info" level in main.rs:108 instead of config file
- RUST_LOG env var override is post-initialization
- For structured logging or log aggregation, requires code changes
- No request IDs or game IDs in move logs - hard to correlate multi-game logs