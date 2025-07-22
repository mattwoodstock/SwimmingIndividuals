
<div style="text-align: center; margin-bottom: 2rem;">
  <h1 style="text-align: center; font-size: 3rem; font-weight: bold; color: #003f5c; margin-bottom: 0.5rem;">SwimmingIndividuals.jl</h1>
  <p style="font-size: 1.25rem; color: #5a5a5a;">
    A High-Performance Agent-Based Model for Marine Ecosystems
  </p>
</div>

## Introduction 

Welcome to the official user guide for **SwimmingIndividuals.jl v1.0**. This documentation provides a comprehensive resource for understanding the model's architecture, configuring and running simulations, and analyzing the output.

`SwimmingIndividuals.jl` is a next-generation modeling framework designed to simulate the life histories and population dynamics of marine organisms. It is built in the high-performance **Julia** programming language and leverages modern parallel computing to run efficiently on both multi-core **CPUs** and **NVIDIA GPUs**.

The model’s core philosophy is to simulate ecosystems from the *bottom-up*, where large-scale patterns in population size, distribution, and structure emerge from the mechanistic, process-based decisions of individual agents.

---

## 🔑 Key Features

::: {.grid}

::: {.g-col-12 .g-col-md-6}
::: {.card}

#### 🧠 **Mechanistic Biology**
Agents are governed by detailed, process-based sub-models for movement, behavior, bioenergetics, and predation.
:::
:::

::: {.g-col-12 .g-col-md-6}
::: {.card}
#### 📊 **Data-Driven**
The model world is built from standard NetCDF and CSV files, allowing for easy integration with real-world environmental data.
:::
:::

::: {.g-col-12 .g-col-md-6}
::: {.card}
#### 🧩 **Flexible & Modular**
The framework is designed to be extensible, allowing users to easily define new species, behaviors, and fishery regulations.
:::
:::

::: {.g-col-12 .g-col-md-6}
::: {.card}
#### ⚡ **High-Performance**
The dual CPU/GPU architecture allows for the simulation of millions of individual agents over large spatial and temporal scales.
:::
:::

:::

---

## 📘 How to Use This Guide

This user guide is organized to walk you through the entire process of using the model from setup to analysis.

Use the **sidebar** to navigate through:

- Input configuration
- Model components
- Simulation setup and execution
- Output and diagnostics
- Example applications

_Last updated: July 22, 2025_
