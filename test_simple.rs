extern crate starter_snake_rust;

fn main() {
    // Test re-exported components
    let voronoi = starter_snake_rust::VoronoiTerritoryAnalyzer::new();
    let danger = starter_snake_rust::DangerZonePredictor::new(3);
    
    println!("SUCCESS: Re-exported components work");
    println!("VoronoiTerritoryAnalyzer: instantiated");
    println!("DangerZonePredictor: instantiated");
}