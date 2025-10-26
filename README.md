#KnightHacks2025
# DronePath

DronePath is a data-driven optimization and visualization tool designed to streamline drone missions for power line 
inspection across Florida. By analyzing flight data, geographical constraints, and energy efficiency, DronePath computes the 
most optimal routes for multiple drones while adhering to mission boundaries and battery limits — **ensuring reliable coverage and smarter resource allocation**.

---

## Inspiration
Our team wanted to take on a real-world challenge that impacts millions — maintaining Florida’s vast power infrastructure. The NextEra Energy Drone Optimization Challenge was a perfect fit, letting us combine computer science, optimization, and sustainability.

We were inspired by:
- The opportunity to enhance energy reliability through drone-based inspections
- The challenge of working with **real-world geospatial and flight data**
- A shared goal of creating smarter, data-driven solutions that make field operations more efficient and scalable
---

## What it does
- Processes and visualizes drone flight data to determine optimal mission routes under real constraints
- Feeds converted data of coords into an **optimization algorithm** to compute efficient flight path 
- Considers constraints like battery life, airspace boundaries, power pole coverage, and mission limits
- Visualizes the resulting optimized routes using **Plotly, displaying total distance traveled, estimated energy usage, and number of missions needed**

---

## How we built it
- **Tech Stack:** Python, NumPy, Google OR-Tools, Shapely, Plotly
- **Tools:** VS Code, GitHub, Git
- **Extra helpers:** Sample datasets provided by the challenge, ChatGPT

---

## Challenges we ran into
- Handling and interpreting .npy files and navigation graph data formats
- Merging branches and maintaining clean collaboration while learning Git workflows
- Debugging mismatched datasets that caused incorrect visualizations early on
- Balancing the complexity of optimization logic with** real-time data visualization**

---

## Accomplishments we're proud of
- Integrating multiple complex datasets into a fully working optimization pipeline
- **Creating an intuitive Plotly visualization** that helps analyze routes and mission efficiency
- Successfully applying graph algorithms like Dijkstra and constraint-based optimization using Google OR-Tools
- Building confidence as a team through effective debugging, version control, and collaboration

---

## What we learned
- Applying graph algorithms (Dijkstra, TSP, greedy methods) to real geospatial problems
- **Using Google OR-Tools** for route optimization under multiple constraints
- Strengthening our Git and collaborative coding skills through feature branches and merge management
- Understanding how data visualization can help communicate technical insights clearly

---

## Installation & Usage
1. Clone and download this repository.  
2. Have the necessary data files and run the main.py file
3. View the missions_all_in_one_sidebyside.html file that's outputted on a live server using npx live-server .
5. **DronePath's visual is presented and interactive **

---

## Team
- **Jesus Gonzalez** – (UI/Backend)  
- **Alvaro Canseco-Martinez** – (UI/Presentation)  
- **Ryan Nyguen** – (UI/Backend)
- **Richard Zhang** - (UI/Backend)
