use rand::Rng;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct City {
    pub x: f32,
    pub y: f32,
}

#[derive(Debug)]
pub struct Ant {
    pub path: Vec<usize>,
    pub tour_length: f32,
    pub visited: Vec<bool>,
}

impl Ant {
    pub fn new(num_cities: usize) -> Ant {
        Ant {
            path: vec![0; num_cities],
            tour_length: 0.0,
            visited: vec![false; num_cities],
        }
    }

    pub fn move_to(&mut self, city: usize, distance: f32) {
        self.path.push(city);
        self.tour_length += distance;
        self.visited[city] = true;
    }

    pub fn visited(&self, city: usize) -> bool {
        self.visited[city]
    }
}

pub struct Aco<'a> {
    pub cities: &'a Vec<City>,
    pub num_ants: usize,
    pub num_iterations: usize,
    pub alpha: f32,
    pub beta: f32,
    pub evaporation_rate: f32,
    pub pheromone_deposit: f32,
    pub pheromones: Vec<Vec<f32>>,
    pub distances: Vec<Vec<f32>>,
    pub best_ant: Ant,
}

impl<'a> Aco<'a> {
    pub fn new(
        cities: &'a Vec<City>,
        num_ants: usize,
        num_iterations: usize,
        alpha: f32,
        beta: f32,
        evaporation_rate: f32,
        pheromone_deposit: f32,
    ) -> Aco<'a> {
        let num_cities = cities.len();
        let mut pheromones = vec![vec![1.0 / (num_cities as f32); num_cities]; num_cities];
        let mut distances = vec![vec![0.0; num_cities]; num_cities];
        for i in 0..num_cities {
            for j in 0..num_cities {
                if i != j {
                    let dx = cities[i].x - cities[j].x;
                    let dy = cities[i].y - cities[j].y;
                    distances[i][j] = (dx * dx + dy * dy).sqrt();
                }
            }
        }

        Aco {
            cities,
            num_ants,
            num_iterations,
            alpha,
            beta,
            evaporation_rate,
            pheromone_deposit,
            pheromones,
            distances,
            best_ant: Ant::new(num_cities),
        }
    }

    pub fn run(&mut self) -> &Ant {
        let num_cities = self.cities.len();
        let mut ants = vec![Ant::new(num_cities); self.num_ants];

        for _ in 0..self.num_iterations {

            for ant in ants.iter_mut() {
                self.move_ant(ant);
    }
        self.update_pheromones(&mut ants);

        let best_ant = ants.iter().max_by(|a, b| a.tour_length.partial_cmp(&b.tour_length).unwrap()).unwrap();
        if best_ant.tour_length < self.best_tour_length {
            self.best_tour_length = best_ant.tour_length;
            self.best_tour = best_ant.tour.clone();
        }

        ants.iter_mut().for_each(|ant| ant.reset());
    }

    self
}

fn move_ant(&self, ant: &mut Ant) {
    let current_city = ant.current_city.unwrap();
    let allowed_cities = self.allowed_cities(ant);

    if allowed_cities.is_empty() {
        return;
    }

    let next_city = if allowed_cities.len() == 1 {
        allowed_cities[0]
    } else {
        let total_pheromone = allowed_cities.iter().map(|&c| self.pheromones[current_city][c]).sum::<f64>();
        let mut probability_sum = 0.0;
        let random_number = self.rng.gen::<f64>();

        for &city in allowed_cities.iter() {
            probability_sum += self.pheromones[current_city][city] / total_pheromone;
            if random_number <= probability_sum {
                break city;
            }
        }

        *allowed_cities.last().unwrap()
    };

    ant.visit_city(next_city, self.distance_matrix[current_city][next_city]);
}

fn allowed_cities(&self, ant: &Ant) -> Vec<usize> {
    let current_city = ant.current_city.unwrap();
    let visited_cities = &ant.visited_cities;
    let allowed_cities = self.cities.iter().enumerate().filter(|&(i, _)| !visited_cities.contains(&i) && i != current_city).map(|(i, _)| i).collect::<Vec<_>>();

    allowed_cities
}

fn update_pheromones(&mut self, ants: &mut [Ant]) {
    for pheromone in self.pheromones.iter_mut().flatten() {
        *pheromone *= self.evaporation_rate;
    }

    for ant in ants.iter() {
        let tour_length = ant.tour_length;
        let tour = &ant.tour;
        for i in 0..(tour.len() - 1) {
            let current_city = tour[i];
            let next_city = tour[i + 1];
            self.pheromones[current_city][next_city] += self.q / tour_length;
            self.pheromones[next_city][current_city] = self.pheromones[current_city][next_city];
        }
    }
}
}

 pub struct Ant {
 visited_cities: HashSet<usize>,
 tour: Vec<usize>,
 tour_length: f64,
 current_city: Option<usize>,
}

  impl Ant {
  fn new(num_cities: usize) -> Self {
  Self {
  visited_cities: HashSet::new(),
  tour: Vec::with_capacity(num_cities),
  tour_length: 0.0,
  current_city: None,
   }
}
  }
fn visit_city(&mut self, city: usize, distance: f64) {
    if let Some(current_city) = self.current_city {
        self.tour_length += distance;
        self.tour.push(city);
        self.visited_cities.insert(current_city);
    }
    self.current_city = Some(city);
}

fn reset(&mut self) {
    self.visited_cities.clear();
    ();
self.current_city = None;
self.tour.clear();
self.tour_length = 0.0;
}

fn calculate_probabilities(&self, ant: &Ant) -> Vec<f64> {
let current_city = ant.current_city.expect("Ant has no current city");
let mut unvisited_cities: Vec<usize> = self
.cities
.iter()
.enumerate()
.filter(|(i, _)| !ant.visited_cities.contains(i))
.map(|(i, _)| i)
.collect();

unvisited_cities.sort_by(|a, b| {
    let pheromone_a = self.pheromone_trails[current_city][*a];
    let pheromone_b = self.pheromone_trails[current_city][*b];
    pheromone_a.partial_cmp(&pheromone_b).unwrap_or(Ordering::Equal)
});

let mut probabilities = vec![0.0; self.num_cities];
let mut total_prob = 0.0;

for city in unvisited_cities {
    let distance = self.distance_matrix[current_city][city];
    let pheromone = self.pheromone_trails[current_city][city];

    probabilities[city] = pheromone.powf(self.alpha) * ((1.0 / distance).powf(self.beta));
    total_prob += probabilities[city];
 }

for prob in probabilities.iter_mut() {
    *prob /= total_prob;
 }

probabilities
}

fn update_pheromone_trails(&mut self, ants: &[Ant]) {
for row in self.pheromone_trails.iter_mut() {
for pheromone in row.iter_mut() {
*pheromone *= 1.0 - self.evaporation_rate;
  }
 }
}
for ant in ants {
    let tour_length = ant.tour_length;
    let tour = ant.tour.as_slice();

    for i in 0..tour.len() - 1 {
        let current_city = tour[i];
        let next_city = tour[i + 1];
        let pheromone = self.pheromone_trails[current_city][next_city];

        self.pheromone_trails[current_city][next_city] =
            pheromone + (self.q / tour_length);
    }

    let first_city = tour[0];
    let last_city = tour[tour.len() - 1];
    let pheromone = self.pheromone_trails[last_city][first_city];

    self.pheromone_trails[last_city][first_city] = pheromone + (self.q / tour_length);
}
#[cfg(test)]
mod tests {
use super::*;
}
#[test]

fn test_tsp_solver() {

    let cities = vec![
        (0.0, 0.0),
        (1.0, 1.0),
        (2.0, 2.0),
        (3.0, 3.0),
        (4.0, 4.0),
        (5.0, 5.0),
        (6.0, 6.0),
    ];
    let solver
}
    {
let mut solver = TspSolver::new(cities, 10, 100);

let best_tour = solver.solve();

let total_length = solver.calculate_tour_length(&best_tour);

println!("Best tour: {:?}", best_tour);
println!("Total tour length: {}", total_length);
}