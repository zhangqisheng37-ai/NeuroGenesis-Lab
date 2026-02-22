use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct TickSnapshot {
    #[pyo3(get)]
    pub tick: usize,
    #[pyo3(get)]
    pub alive_neurons: usize,
    #[pyo3(get)]
    pub total_neurons: usize,
    #[pyo3(get)]
    pub tick_spikes: usize,
    #[pyo3(get)]
    pub propagation_count: usize,
    #[pyo3(get)]
    pub mean_energy: f64,
}

#[pymethods]
impl TickSnapshot {
    #[new]
    pub fn new(
        tick: usize,
        alive_neurons: usize,
        total_neurons: usize,
        tick_spikes: usize,
        propagation_count: usize,
        mean_energy: f64,
    ) -> Self {
        Self {
            tick,
            alive_neurons,
            total_neurons,
            tick_spikes,
            propagation_count,
            mean_energy,
        }
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("tick", self.tick)?;
        dict.set_item("alive_neurons", self.alive_neurons)?;
        dict.set_item("total_neurons", self.total_neurons)?;
        dict.set_item("tick_spikes", self.tick_spikes)?;
        dict.set_item("propagation_count", self.propagation_count)?;
        dict.set_item("mean_energy", self.mean_energy)?;
        Ok(dict)
    }
}

#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct EngineSummary {
    #[pyo3(get)]
    pub total_neurons: usize,
    #[pyo3(get)]
    pub alive_neurons: usize,
    #[pyo3(get)]
    pub total_spikes: usize,
    #[pyo3(get)]
    pub total_propagations: usize,
    #[pyo3(get)]
    pub mean_energy: f64,
}

#[pymethods]
impl EngineSummary {
    #[new]
    pub fn new(
        total_neurons: usize,
        alive_neurons: usize,
        total_spikes: usize,
        total_propagations: usize,
        mean_energy: f64,
    ) -> Self {
        Self {
            total_neurons,
            alive_neurons,
            total_spikes,
            total_propagations,
            mean_energy,
        }
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("total_neurons", self.total_neurons)?;
        dict.set_item("alive_neurons", self.alive_neurons)?;
        dict.set_item("total_spikes", self.total_spikes)?;
        dict.set_item("total_propagations", self.total_propagations)?;
        dict.set_item("mean_energy", self.mean_energy)?;
        Ok(dict)
    }
}

#[pyclass]
pub struct LIFCore {
    #[pyo3(get)]
    pub v: f64,
    #[pyo3(get)]
    pub v_rest: f64,
    #[pyo3(get)]
    pub v_reset: f64,
    #[pyo3(get)]
    pub v_th: f64,
    #[pyo3(get)]
    pub tau_m: f64,
    #[pyo3(get)]
    pub r_m: f64,
}

#[pymethods]
impl LIFCore {
    #[new]
    pub fn new(v_rest: f64, v_reset: f64, v_th: f64, tau_m: f64, r_m: f64) -> Self {
        Self {
            v: v_rest,
            v_rest,
            v_reset,
            v_th,
            tau_m,
            r_m,
        }
    }

    pub fn tick(&mut self, input_current: f64, dt: f64) -> PyResult<bool> {
        if dt <= 0.0 {
            return Err(PyValueError::new_err("dt must be positive"));
        }
        if self.tau_m <= 0.0 {
            return Err(PyValueError::new_err("tau_m must be positive"));
        }

        let dv = (-(self.v - self.v_rest) + self.r_m * input_current) / self.tau_m * dt;
        self.v += dv;

        if self.v >= self.v_th {
            self.v = self.v_reset;
            return Ok(true);
        }
        Ok(false)
    }
}

#[pyclass]
pub struct IzhikevichCore {
    #[pyo3(get)]
    pub v: f64,
    #[pyo3(get)]
    pub u: f64,
    #[pyo3(get)]
    pub a: f64,
    #[pyo3(get)]
    pub b: f64,
    #[pyo3(get)]
    pub c: f64,
    #[pyo3(get)]
    pub d: f64,
}

#[pymethods]
impl IzhikevichCore {
    #[new]
    pub fn new(a: f64, b: f64, c: f64, d: f64) -> Self {
        let v0 = -65.0;
        let u0 = b * v0;
        Self {
            v: v0,
            u: u0,
            a,
            b,
            c,
            d,
        }
    }

    pub fn tick(&mut self, input_current: f64, dt: f64) -> PyResult<bool> {
        if dt <= 0.0 {
            return Err(PyValueError::new_err("dt must be positive"));
        }

        let dv = (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + input_current) * dt;
        let du = (self.a * (self.b * self.v - self.u)) * dt;
        self.v += dv;
        self.u += du;

        if self.v >= 30.0 {
            self.v = self.c;
            self.u += self.d;
            return Ok(true);
        }
        Ok(false)
    }
}

#[pyclass]
pub struct SparseAdjacency {
    adjacency: Vec<Vec<(usize, f64, usize)>>,
}

#[pymethods]
impl SparseAdjacency {
    #[new]
    pub fn new(node_count: usize) -> Self {
        Self {
            adjacency: vec![Vec::new(); node_count],
        }
    }

    pub fn connect(&mut self, src: usize, dst: usize, weight: f64, delay: usize) -> PyResult<()> {
        if src >= self.adjacency.len() || dst >= self.adjacency.len() {
            return Err(PyValueError::new_err("src/dst out of range"));
        }
        self.adjacency[src].push((dst, weight, delay));
        Ok(())
    }

    pub fn neighbors(&self, src: usize) -> PyResult<Vec<(usize, f64, usize)>> {
        if src >= self.adjacency.len() {
            return Err(PyValueError::new_err("src out of range"));
        }
        Ok(self.adjacency[src].clone())
    }

    pub fn size(&self) -> usize {
        self.adjacency.len()
    }
}

#[pyclass]
pub struct EventQueue {
    events: Vec<(usize, usize, usize, f64)>,
}

#[pymethods]
impl EventQueue {
    #[new]
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    pub fn push(&mut self, tick: usize, src: usize, dst: usize, weight: f64) {
        self.events.push((tick, src, dst, weight));
    }

    pub fn pop_ready(&mut self, current_tick: usize) -> Vec<(usize, usize, usize, f64)> {
        let mut ready = Vec::new();
        let mut pending = Vec::new();
        for item in self.events.drain(..) {
            if item.0 <= current_tick {
                ready.push(item);
            } else {
                pending.push(item);
            }
        }
        self.events = pending;
        ready
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }
}

#[pyfunction]
pub fn parallel_lif_tick(
    voltages: Vec<f64>,
    v_rest: f64,
    v_reset: f64,
    v_th: f64,
    tau_m: f64,
    r_m: f64,
    inputs: Vec<f64>,
    dt: f64,
) -> PyResult<(Vec<f64>, Vec<bool>)> {
    if voltages.len() != inputs.len() {
        return Err(PyValueError::new_err("voltages and inputs length mismatch"));
    }
    if dt <= 0.0 || tau_m <= 0.0 {
        return Err(PyValueError::new_err("dt and tau_m must be positive"));
    }

    let mut out_v = voltages;
    let mut spikes = vec![false; out_v.len()];

    out_v
        .par_iter_mut()
        .zip(spikes.par_iter_mut())
        .zip(inputs.par_iter())
        .for_each(|((v, spike), input)| {
            let dv = (-(*v - v_rest) + r_m * *input) / tau_m * dt;
            *v += dv;
            if *v >= v_th {
                *v = v_reset;
                *spike = true;
            }
        });

    Ok((out_v, spikes))
}

#[pyfunction]
pub fn mine_pattern_counts(
    samples: Vec<Vec<i64>>,
    min_support: usize,
) -> PyResult<(Vec<(i64, i64, i64, usize)>, Vec<(i64, i64, usize)>)> {
    if min_support == 0 {
        return Err(PyValueError::new_err("min_support must be >= 1"));
    }

    let mut trigram_support: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let mut bigram_support: HashMap<(i64, i64), usize> = HashMap::new();

    for sequence in samples {
        let mut trigrams: HashSet<(i64, i64, i64)> = HashSet::new();
        let mut bigrams: HashSet<(i64, i64)> = HashSet::new();

        if sequence.len() >= 3 {
            for window in sequence.windows(3) {
                trigrams.insert((window[0], window[1], window[2]));
            }
        }
        if sequence.len() >= 2 {
            for window in sequence.windows(2) {
                bigrams.insert((window[0], window[1]));
            }
        }

        for trigram in trigrams {
            *trigram_support.entry(trigram).or_insert(0) += 1;
        }
        for bigram in bigrams {
            *bigram_support.entry(bigram).or_insert(0) += 1;
        }
    }

    let trigrams = trigram_support
        .into_iter()
        .filter_map(|((a, b, c), support)| {
            if support >= min_support {
                Some((a, b, c, support))
            } else {
                None
            }
        })
        .collect();

    let bigrams = bigram_support
        .into_iter()
        .filter_map(|((a, b), support)| {
            if support >= min_support {
                Some((a, b, support))
            } else {
                None
            }
        })
        .collect();

    Ok((trigrams, bigrams))
}

#[pymodule]
fn _neurogenesis_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TickSnapshot>()?;
    m.add_class::<EngineSummary>()?;
    m.add_class::<LIFCore>()?;
    m.add_class::<IzhikevichCore>()?;
    m.add_class::<SparseAdjacency>()?;
    m.add_class::<EventQueue>()?;
    m.add_function(wrap_pyfunction!(parallel_lif_tick, m)?)?;
    m.add_function(wrap_pyfunction!(mine_pattern_counts, m)?)?;
    Ok(())
}
