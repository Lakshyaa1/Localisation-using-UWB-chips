# UWB Localization using Qorvo DWM3001C

This repository contains experiments and tools for **Ultra-Wideband (UWB) based indoor localization** using **Qorvo DWM3001C development kits**.

The project demonstrates **centimeter-level positioning using UWB ranging**, along with **Python-based trilateration and filtering algorithms** for estimating tag position.

The repository includes:

- Scripts for **position estimation and filtering**
- Data collected from **multiple localization experiments**
- Configuration steps for running **Qorvo’s official firmware**

This setup achieved **~3 cm localization accuracy** under controlled conditions.

---

# Ultra-Wideband (UWB) Localization

Ultra-Wideband (UWB) is a wireless communication technology that uses **very short radio pulses across a wide bandwidth**.

Unlike traditional wireless technologies, UWB can measure **time-of-flight of signals with extremely high precision**, allowing accurate distance estimation.

Since radio signals travel approximately:

```
30 cm per nanosecond
```

precise timestamp measurements allow **centimeter-level ranging accuracy**.

In this project, localization is performed using **UWB ranging between a tag and multiple anchors**. The distances from the anchors are then used to estimate the tag position using **trilateration**.

---

# Hardware

This project uses:

- **Qorvo DWM3001C Development Kits**
- Multiple **anchor nodes**
- One **tag node**

The anchors measure their distance to the tag using UWB ranging.

These distance measurements are then processed using Python scripts to estimate the tag's position.

---

# SDK Setup

Download the **Qorvo DWM3001C SDK** from:

https://www.qorvo.com/products/p/DWM3001CDK#documents

After downloading the SDK:

1. Build or obtain the **CLI firmware** provided in the SDK.
2. Flash the firmware onto each **DWM3001C board**.

Once flashed, the boards can be controlled using the **serial CLI interface**.

---

# Connecting to the Board

Each board exposes a **UART interface through the J20 port**.

To communicate with the board:

1. Connect the board via USB
2. Open the serial terminal using **minicom**

Example:

```
minicom -D /dev/ttyACM0
```

Once connected, you can issue **CLI commands to configure the nodes**.

---

# Node Configuration

In the localization setup:

- **One device acts as the tag (initiator)**
- **Multiple devices act as anchors (responders)**

---

## Tag Configuration (Initiator)

The tag initiates the ranging process.

Example command:

```
initf -multi -addr=1 -paddr=[0,2,3,4,6,7,9,10]
```

Explanation:

- `initf` → initiator mode  
- `-multi` → enable multi-anchor ranging  
- `-addr=1` → address of the tag node  
- `-paddr=[...]` → list of anchor addresses participating in ranging  

The values inside `paddr` represent the **addresses of the anchors**.

---

## Anchor Configuration (Responder)

Each anchor responds to the tag's ranging requests.

For each anchor:

- `addr` = anchor address
- `paddr` = address of the initiator (tag)

Example for anchor **0**:

```
respf -multi -addr=0 -paddr=1
```

Explanation:

- `respf` → responder mode  
- `-multi` → multi-node operation  
- `-addr=0` → anchor address  
- `-paddr=1` → address of the tag  

Each anchor should be configured similarly with its **own address**.

---

# Data Collection

During experiments, distance measurements between the tag and anchors were collected.

These datasets are stored in the repository and correspond to **different localization setups and anchor placements**.

The data is used for evaluating localization performance and testing different filtering methods.

---

# Python Trilateration Scripts

The `scripts/` folder contains Python implementations for **position estimation and filtering**.

These scripts process the distance measurements obtained from the UWB ranging system.

Implemented techniques include:

- **Trilateration algorithms**
- **Nonlinear Least Squares (NLLS) optimization**
- **Exponential smoothing filters**
- **Robust loss functions**

These filters help improve localization accuracy by reducing the effect of:

- measurement noise
- multipath interference
- outlier distance estimates

---

# Repository Structure

```
scripts/
    Python scripts for trilateration and filtering

data/
    Collected datasets from localization experiments

docs/
    Additional notes and experiment details
```

---

# Results

Using the setup described in this repository, we achieved:

- **~3 cm localization accuracy**
- Stable multi-anchor ranging
- Successful trilateration using Python-based solvers

These experiments served as the **foundation for further development of custom firmware** in the Zephyr-based implementation.

---

# Related Work

Custom firmware implementation based on **Zephyr RTOS**:
(https://github.com/vedantmalkar/zephyr-dw3001cdk-tdoa.git)
