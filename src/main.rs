#[macro_use]
extern crate rocket;

use log::info;
use rocket::fairing::AdHoc;
use rocket::http::Status;
use rocket::serde::{json::Json, Deserialize};
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::env;
use std::time::{SystemTime, UNIX_EPOCH};

mod logic;

// API and Response Objects
// See https://docs.battlesnake.com/api

#[derive(Deserialize, Serialize, Debug)]
pub struct Game {
    id: String,
    ruleset: HashMap<String, Value>,
    timeout: u32,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct Board {
    height: u32,
    width: i32,
    food: Vec<Coord>,
    snakes: Vec<Battlesnake>,
    hazards: Vec<Coord>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Battlesnake {
    pub id: String,
    pub name: String,
    pub health: i32,
    pub body: Vec<Coord>,
    pub head: Coord,
    pub length: i32,
    pub latency: String,
    pub shout: Option<String>,
}

#[derive(Deserialize, Serialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Coord {
    pub x: i32,
    pub y: i32,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct GameState {
    game: Game,
    turn: i32,
    board: Board,
    you: Battlesnake,
}

#[get("/")]
fn handle_index() -> Json<Value> {
    Json(logic::info())
}

#[post("/start", format = "json", data = "<start_req>")]
fn handle_start(start_req: Json<GameState>) -> Status {
    logic::start(
        &start_req.game,
        &start_req.turn,
        &start_req.board,
        &start_req.you,
    );

    Status::Ok
}

#[post("/move", format = "json", data = "<move_req>")]
fn handle_move(move_req: Json<GameState>) -> Json<Value> {
    // Generate unique request ID for tracking
    let request_start = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let request_id = format!("{}-{}", request_start, move_req.turn);
    
    info!("API REQUEST START: {} - Turn: {}, Snake: {} at ({}, {})",
          request_id, move_req.turn, move_req.you.id, move_req.you.head.x, move_req.you.head.y);

    let response = logic::get_move(
        &move_req.game,
        &move_req.turn,
        &move_req.board,
        &move_req.you,
    );

    // CRITICAL DEBUGGING: Add response validation and logging
    let response_end = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let response_time = response_end - request_start;
    
    let response_json = serde_json::to_string(&response).unwrap_or_default();
    info!("API RESPONSE COMPLETE: {} - Processing time: {}ms, Raw response JSON: {}",
          request_id, response_time, response_json);
    
    // Validate response structure
    if let Some(move_field) = response.get("move") {
        let move_str = move_field.as_str().unwrap_or("invalid");
        info!("API RESPONSE VALIDATION: {} - Move field: '{}', Type: {}",
              request_id, move_str, move_field);
        
        // Check for expected move values
        match move_str {
            "up" | "down" | "left" | "right" => {
                info!("API RESPONSE SUCCESS: {} - Valid move '{}' detected", request_id, move_str);
            },
            _ => {
                info!("API RESPONSE ERROR: {} - Invalid move value: '{}'", request_id, move_str);
            }
        }
    } else {
        info!("API RESPONSE ERROR: {} - No 'move' field found in response", request_id);
        info!("API RESPONSE ERROR: {} - Available fields: {:?}",
              request_id, response.as_object().map(|obj| obj.keys().collect::<Vec<_>>()).unwrap_or_default());
    }

    Json(response)
}

#[post("/end", format = "json", data = "<end_req>")]
fn handle_end(end_req: Json<GameState>) -> Status {
    logic::end(&end_req.game, &end_req.turn, &end_req.board, &end_req.you);

    Status::Ok
}

#[launch]
fn rocket() -> _ {
    // Lots of web hosting services expect you to bind to the port specified by the `PORT`
    // environment variable. However, Rocket looks at the `ROCKET_PORT` environment variable.
    // If we find a value for `PORT`, we set `ROCKET_PORT` to that value.
    if let Ok(port) = env::var("PORT") {
        env::set_var("ROCKET_PORT", &port);
    }

    // We default to 'info' level logging. But if the `RUST_LOG` environment variable is set,
    // we keep that value instead.
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "info");
    }

    env_logger::init();

    info!("Starting Battlesnake Server...");

    rocket::build()
        .attach(AdHoc::on_response("Server ID Middleware", |_, res| {
            Box::pin(async move {
                res.set_raw_header("Server", "battlesnake/github/starter-snake-rust");
            })
        }))
        .mount(
            "/",
            routes![handle_index, handle_start, handle_move, handle_end],
        )
}
