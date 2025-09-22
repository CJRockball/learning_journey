# Patrick Carlberg – Data Science Learning Journey (2020–2025)

> **Transforming a career gap into deep technical expertise, project-based learning, and modern AI pipelines.**
<br>


## Executive Summary

This repository documents my systematic journey from domain expert in nanotechnology to full-stack, production-ready data scientist. Over five years, I completed 70+ hands-on projects, 50+ advanced certifications, and delivered modern machine learning solutions using the latest open-source libraries and cloud platforms.

- **Foundation Building:** Intensive upskilling in Python, statistics, ML, and cloud
- **Practical Application:** End-to-end projects from data wrangling to web deployment
- **Modern Workflows:** Productionization via Docker, FastAPI, LangChain, and more
- **Learning-in-Public:** Transparent record of raw code, refactoring, and skill development

All raw project code is documented *as is* to reflect genuine skill progression. Folders are organized to highlight learning phases and technology stacks, with honest badges indicating code maturity.
<br>

## Interactive Project Timeline

<div id="timeline-container" style="width: 100%; height: 600px; border: 1px solid #ddd; margin: 20px 0;"></div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
// Project timeline data extracted from your actual timeline
const timelineData = [
  // 2021 - Foundation Phase
  {date: "2021-01-29", title: "Online Stock DB Graphs", category: "web", phase: "foundations", duration: 5},
  {date: "2021-03-04", title: "IEEE Fraud Dataset", category: "ml", phase: "foundations", duration: 9},
  {date: "2021-04-01", title: "TensorFlow Coursera", category: "dl", phase: "foundations", duration: 60},
  {date: "2021-06-10", title: "Reinforcement Learning", category: "ml", phase: "foundations", duration: 11},
  {date: "2021-07-15", title: "Bayesian Statistics", category: "stats", phase: "foundations", duration: 31},
  {date: "2021-09-23", title: "Million Song XGB", category: "ml", phase: "foundations", duration: 5},
  {date: "2021-10-01", title: "Flask Heroku Deploy", category: "web", phase: "foundations", duration: 11},
  
  // 2022 - Applied ML Phase
  {date: "2022-01-06", title: "FastAPI Dashboard", category: "web", phase: "applied", duration: 21},
  {date: "2022-02-14", title: "Mandelbrot GPU", category: "gpu", phase: "applied", duration: 69},
  {date: "2022-03-19", title: "Option Pricing GPU", category: "finance", phase: "applied", duration: 5},
  {date: "2022-08-15", title: "PyTorch MNIST/GAN", category: "dl", phase: "applied", duration: 7},
  {date: "2022-11-02", title: "CitiBike Big Data (28GB)", category: "bigdata", phase: "applied", duration: 26},
  
  // 2023 - Production Phase
  {date: "2023-01-28", title: "Large Timeseries XGB", category: "ml", phase: "production", duration: 54},
  {date: "2023-04-04", title: "FastAPI Microservices", category: "web", phase: "production", duration: 11},
  {date: "2023-07-20", title: "NLP Book Project", category: "nlp", phase: "production", duration: 37},
  {date: "2023-09-08", title: "Singa-Rent Full Stack", category: "web", phase: "production", duration: 35},
  {date: "2023-10-13", title: "CUDA Programming", category: "gpu", phase: "production", duration: 2},
  
  // 2024 - Modern AI Phase
  {date: "2024-03-21", title: "Time Series Analysis", category: "stats", phase: "modern", duration: 16},
  {date: "2024-06-19", title: "PyTorch RNN Timeseries", category: "dl", phase: "modern", duration: 4},
  {date: "2024-07-14", title: "Kaggle PyTorch Competition", category: "competition", phase: "modern", duration: 17},
  {date: "2024-08-26", title: "NeurIPS Ariel Challenge", category: "competition", phase: "modern", duration: 11},
  {date: "2024-10-17", title: "Jane Street Competition", category: "finance", phase: "modern", duration: 15},
  {date: "2024-12-31", title: "CIBMTR Survival Models", category: "ml", phase: "modern", duration: 18},
  
  // 2025 - Current
  {date: "2025-03-25", title: "LangChain/LangGraph", category: "ai", phase: "modern", duration: 8},
  {date: "2025-07-04", title: "Streamlit RAG Chat", category: "ai", phase: "modern", duration: 10},
  {date: "2025-07-18", title: "FastAPI ML Serving", category: "web", phase: "modern", duration: 4}
];

// Categories and colors
const categoryColors = {
  "web": "#3498db",
  "ml": "#e74c3c", 
  "dl": "#9b59b6",
  "stats": "#f39c12",
  "gpu": "#2ecc71",
  "finance": "#1abc9c",
  "bigdata": "#34495e",
  "nlp": "#e67e22",
  "competition": "#8e44ad",
  "ai": "#ff6b6b"
};

const phaseColors = {
  "foundations": "#74b9ff",
  "applied": "#00cec9", 
  "production": "#fdcb6e",
  "modern": "#fd79a8"
};

// Set up dimensions
const container = document.getElementById('timeline-container');
const margin = {top: 50, right: 150, bottom: 50, left: 50};
const width = container.offsetWidth - margin.left - margin.right;
const height = 550 - margin.top - margin.bottom;

// Create SVG
const svg = d3.select('#timeline-container')
  .append('svg')
  .attr('width', width + margin.left + margin.right)
  .attr('height', height + margin.top + margin.bottom);

const g = svg.append('g')
  .attr('transform', `translate(${margin.left},${margin.top})`);

// Parse dates
const parseDate = d3.timeParse("%Y-%m-%d");
timelineData.forEach(d => {
  d.date = parseDate(d.date);
});

// Set up scales
const xScale = d3.scaleTime()
  .domain(d3.extent(timelineData, d => d.date))
  .range([0, width]);

const yScale = d3.scaleBand()
  .domain(['foundations', 'applied', 'production', 'modern'])
  .range([0, height])
  .padding(0.2);

// Create timeline axis
const xAxis = d3.axisBottom(xScale)
  .tickFormat(d3.timeFormat("%Y"));

g.append('g')
  .attr('class', 'x-axis')
  .attr('transform', `translate(0,${height})`)
  .call(xAxis);

// Add phase labels
g.selectAll('.phase-label')
  .data(['foundations', 'applied', 'production', 'modern'])
  .enter()
  .append('text')
  .attr('class', 'phase-label')
  .attr('x', -10)
  .attr('y', d => yScale(d) + yScale.bandwidth()/2)
  .attr('dy', '0.35em')
  .style('text-anchor', 'end')
  .style('font-weight', 'bold')
  .style('font-size', '12px')
  .text(d => d.charAt(0).toUpperCase() + d.slice(1));

// Add phase background bars
g.selectAll('.phase-bg')
  .data(['foundations', 'applied', 'production', 'modern'])
  .enter()
  .append('rect')
  .attr('class', 'phase-bg')
  .attr('x', 0)
  .attr('y', d => yScale(d))
  .attr('width', width)
  .attr('height', yScale.bandwidth())
  .style('fill', d => phaseColors[d])
  .style('opacity', 0.1);

// Create tooltip
const tooltip = d3.select('body').append('div')
  .attr('class', 'tooltip')
  .style('position', 'absolute')
  .style('background', 'rgba(0,0,0,0.8)')
  .style('color', 'white') 
  .style('padding', '10px')
  .style('border-radius', '5px')
  .style('pointer-events', 'none')
  .style('opacity', 0);

// Add project circles
const projects = g.selectAll('.project')
  .data(timelineData)
  .enter()
  .append('circle')
  .attr('class', 'project')
  .attr('cx', d => xScale(d.date))
  .attr('cy', d => yScale(d.phase) + yScale.bandwidth()/2)
  .attr('r', d => Math.max(3, Math.min(8, d.duration/3)))
  .style('fill', d => categoryColors[d.category])
  .style('opacity', 0.8)
  .style('cursor', 'pointer')
  .on('mouseover', function(event, d) {
    tooltip.transition().duration(200).style('opacity', .9);
    tooltip.html(`
      <strong>${d.title}</strong><br/>
      Date: ${d3.timeFormat("%B %Y")(d.date)}<br/>
      Duration: ${d.duration} days<br/>
      Category: ${d.category}<br/>
      Phase: ${d.phase}
    `)
    .style('left', (event.pageX + 10) + 'px')
    .style('top', (event.pageY - 28) + 'px');
    
    d3.select(this)
      .transition()
      .duration(100)
      .attr('r', d => Math.max(5, Math.min(12, d.duration/2)))
      .style('opacity', 1);
  })
  .on('mouseout', function(d) {
    tooltip.transition().duration(500).style('opacity', 0);
    d3.select(this)
      .transition()
      .duration(100)
      .attr('r', d => Math.max(3, Math.min(8, d.duration/3)))
      .style('opacity', 0.8);
  });

// Add legend
const legend = svg.append('g')
  .attr('class', 'legend')
  .attr('transform', `translate(${width + margin.left + 10}, ${margin.top})`);

const categories = Object.keys(categoryColors);
const legendItems = legend.selectAll('.legend-item')
  .data(categories)
  .enter()
  .append('g')
  .attr('class', 'legend-item')
  .attr('transform', (d, i) => `translate(0, ${i * 20})`);

legendItems.append('circle')
  .attr('cx', 5)
  .attr('cy', 0)
  .attr('r', 5)
  .style('fill', d => categoryColors[d]);

legendItems.append('text')
  .attr('x', 15)
  .attr('y', 0)
  .attr('dy', '0.35em')
  .style('font-size', '11px')
  .text(d => d);

// Add animation on load
projects
  .style('opacity', 0)
  .transition()
  .duration(50)
  .delay((d, i) => i * 100)
  .style('opacity', 0.8);

</script>
<br>
## Repository Structure

```shell
learning-journey/
│
├── README.md                    # Executive summary and navigation
├── TIMELINE.md                  # Chronological list of 70+ projects
├── Detailed Development.md      # Projects sorted by library or subject 
├── coursera_certificates/       # Folder for certificate PDFs
│   └── README.md                # Summary of certifications
│
├── 01_foundations/              # Phase 1 – Python, Stats, Early ML (2020–2021)
├── 02_machine_learning/         # Phase 2 – Core ML, XGB, PyTorch, TF (2021–2022)
├── 03_web_deployment/           # Phase 3 – Flask/FastAPI, APIs, web apps (2021–2023)
├── 04_advanced_computing/       # GPU/CUDA, Big Data, Optimization (2022–2024)
├── 05_quantitative_finance/     # Financial ML, RL, time series (2020–2024)
├── 06_kaggle_competitions/      # Competitive ML & public benchmarks (2021–2025)
├── 07_modern_ml_ai/             # Modern AI (LLMs, LangChain, RAG, NLP) (2024–2025)
├── 08_portfolio_showcase/       # Presentable, end-to-end project demos
```
<br>


## Phases of Development

**Phase 1: Foundations (2020–2021)**
- Upgraded Python, statistics, and data visualization skills
- Completed foundations via Coursera specializations

**Phase 2: Applied ML (2021–2022)**
- Built and iterated on Kaggle, UCI, and public datasets
- First production ML deployments

**Phase 3: Deployment + Performance (2021–2024)**
- Moved projects to web, experimented with containerization
- GPU, big-data, and scalable ML implementations

**Phase 4: Modern AI (2024–2025)**
- Integrated LLMs, LangChain, generative AI
- Competed in advanced Kaggle and NeurIPS challenges
<br>

## Raw/Unpolished Code & Growth Mindset

Some code in early folders is left intentionally raw or only lightly refactored. This is *deliberate*: it documents practical learning cycles, iterative improvements, and technological catch-up after a career pivot.

See [TIMELINE.md](./TIMELINE.md) for project-by-project progress, with major milestones and evolving code quality tagged along the way.
<br>


## Quick Links

- [Complete Project Timeline](./TIMELINE.md): All projects, with dates and themes
- [Group Timeline – by Tech/Library](./Make%20a%20md%20file.md)
- [Coursera Certificates (Summary)](./coursera_certificates/README.md)
- [Best Portfolio Projects](./08_portfolio_showcase/)
- [GitHub Profile](https://github.com/CJRockball)
<br>

## Value Proposition

- **Self-driven, systematic skill acquisition from first principles to production**
- **End-to-end project delivery, with honesty about unfinished/raw work**
- **Strong documentation practices, even for learning-phase code**

---

**Contact:** [Your email] • [LinkedIn] • [https://github.com/CJRockball](https://github.com/CJRockball)
