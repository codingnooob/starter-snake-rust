# Project Coding Rules (Non-Obvious Only)

## Type System Gotchas
- Board.width is i32, Board.height is u32 (src/main.rs:27-28) - inconsistent by design
- When implementing bounds checking, cast appropriately to avoid type errors
- All Coord fields (x, y) are i32 - use this for calculations

## Battlesnake-Specific Patterns
- Snake body is Vec<Coord> where [0]=head, [1]=neck, [...]=body segments in order
- Movement logic must check body[0] vs body[1] to prevent backward moves (src/logic.rs:59-60)
- Never access body[1] without checking body.len() >= 2 first

## Random Selection
- Project uses rand::rng() without seeding (src/logic.rs:94)
- For testing: inject seeded RNG instead of using global rng()
- Current pattern: `safe_moves.choose(&mut rand::rng()).unwrap()`

## Environment Configuration
- PORT env var is manually translated to ROCKET_PORT in main.rs:101-103
- This is Battlesnake deployment convention, not standard Rocket behavior
- Don't rely on ROCKET_PORT directly - set PORT instead