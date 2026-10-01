Problem Statement
HOW MIGHT WE BUILD A TOOL THAT PREDICTS HOW FLIGHT CHANGES AFFECT HOTEL VISITS IN ABU DHABI?
Problem statement visual
 
Most international visitors arrive in Abu Dhabi by air – but only a share of the people flying in actually end up staying in the city's hotels. Many are just connecting through Abu Dhabi's airport to somewhere else, are residents coming home, or are visiting family and friends. So flights and hotel demand are connected, but the link is indirect: what really matters is how many arriving passengers turn into hotel guests, and that share changes by country and by season.
 
Today, flights and hotels are planned by two separate teams using two separate sets of numbers, with no model that links them. That makes simple "what if" questions surprisingly hard to answer. For example, what happens to hotel guest numbers if:
 
    A new flight route opens from a country that didn't fly here before?
    An airline runs more (or fewer) flights per week on a route?
    Airlines add more seats – through larger aircraft or extra flights?
    A higher (or lower) share of those seats actually gets filled?
    The mix of countries flying in shifts with the seasons?
 
Right now there's no easy way to test any of this and see the effect.We're looking for an interactive scenario simulator that connects flight data to hotel demand. A user should be able to adjust any one of these factors and instantly see how it changes hotel guest numbers – broken down by country of origin and time of year – and see which factors move the number most.
 
There's also a data catch to solve along the way: the two datasets don't define "country" the same way. In the flight data, "country" means where the flight departed from; in the hotel data, "country" means the guest's nationality. These often aren't the same.
Context & Impact
why this problem statement
is actual
 
Nearly every international visitor arrives by air, and a portion of them go on to stay in hotels. So decisions about new flight routes, airline deals, seasonal planning, and hotel capacity are all connected in real life – but not on paper. Today, they're based on separate forecasts and rough estimates, not on a model that actually links flights to hotel stays.
 
If you build that link, big strategic questions suddenly become quick to answer. For example:
 
    How many extra hotel nights would a new twice-weekly flight from a given country generate?
    How much would winter hotel occupancy suffer if an airline reduced its flights?
    Which countries bring in the most hotel guests per available seat?
 
The impact is direct – a tool like this would lead to smarter conversations about which flight routes to pursue, earlier warnings when demand might drop, and better planning for hotels, staffing, and events across the city.
Success Criteria
FOR
APPLICATIONS
01
A working simulator, not a static report
 
The user can change aviation and market factors and see hotel guest numbers update.
02
A transparent conversion chain
 
Seats → passengers → visitors → hotel guest nights, with every assumption visible and adjustable.
03
Validation against history
 
Projections are back-tested on held-out periods and reported with an honest error metric (e.g. WMAPE), not just asserted.
04
Sensitivity analysis
 
A clear answer to which factors move hotel guest numbers most.
05
Useful granularity
 
Results broken down by source market and season, not a single headline number.
06
Decision relevance
 
A short, non-technical explanation of what a DCT Abu Dhabi planner should do differently as a result.
07
Reproducibility
 
Documented code, stated assumptions and clearly acknowledged limitations.
Context & Impact
background information
The conversion chain
 
The core of this challenge is modelling the chain that links air supply to hotel demand:
 
Scheduled seats → arriving passengers (via load factors) → inbound visitors (excluding transfer and transit passengers, adjusting for purpose of travel) → hotel guest nights (hotel-capture rate × average length of stay).
 
Each step can be estimated from historical data and then exposed as an adjustable lever so that users can run scenarios.
Suggested modelling approach
 
Teams are encouraged to combine two layers rather than relying on either alone:
 
    A transparent parametric layer that makes the chain explicit, auditable and explainable to a non-technical stakeholder.
    A machine-learning layer (e.g. gradient boosting or regression) that captures seasonality, source-market effects and non-linear relationships instead of fixed ratios.
 
Factors worth exposing as levers:
 
New or discontinued routes; weekly frequency; aircraft type and seat capacity; load factor; origin market and market mix; month and season; event calendars and public holidays in source markets; transfer / transit share; average length of stay.
Indicative tech stack (guidance, not a requirement)
 
    Data and modelling: Python, pandas or Polars, DuckDB; statsmodels for an econometric baseline; LightGBM or XGBoost for the conversion layer.
    Simulation and sensitivity: A lightweight parametric scenario engine (SimPy optionally for staged flows); SALib for global sensitivity analysis.
    Application and visualisation: Streamlit, or FastAPI with a simple front end; Plotly for charts and side-by-side scenario comparison.
    Packaging: Docker for a reproducible, one-command submission.
 
Ground rules
 
    All data is aggregated; no personal or passenger-level data is involved or permitted.
    State assumptions explicitly and show uncertainty wherever possible — an honest range is worth more than a confident single number.
    Data provided is for use within this competition only.
 
What we will provide
you’ll gain access to …
 
Participating teams will be given a curated data pack covering the aviation and hospitality sides of the challenge:
 
    Flight schedule and capacity data: Scheduled flights by route and date, origin airport and source market, weekly frequency, airline, aircraft type, seat capacity, and connection / transfer indicators. This is the supply side of the chain.
    Supporting reference data: Event calendars, public holidays in key source markets, and a base seasonality profile.
    Passenger traffic data: Actual passenger movements and throughput at Abu Dhabi airports by route and period. This is the realised side, used to derive load factors and validate the seats-to-passengers step.
    A data dictionary: Field-level descriptions, units and known caveats for every dataset provided.
    Hotel performance data: Hotel guest arrivals, guest nights, occupancy and average length of stay by month and by source market.
    Mentorship: Access to DCT Abu Dhabi data scientists and domain experts for guidance during the challenge.
 
Please note: No baseline model, starter notebook or pre-joined table will be provided. Teams are expected to explore, join and prepare the data themselves as part of the challenge. All datasets are supplied in aggregated form – no personal or passenger-level records are included – and are licensed for use within this competition only.
Objectives
This challenge aims to …
01
 
Encourage students to solve real-world engineering problems
02
 
Promote innovation using AI and technology
03
 
Develop environmentally responsible solutions
04
 
Turn aviation and tourism data into a decision-support tool for hotel demand planning
05
 
Build explainable models that a non-technical planner can trust and act on
what you’ll submit
expected outcomes
 
By the end of the competition, each team should upload a 10-pages PDF presentation and a 1–2 minute video of the developed solution / digital prototype, including:
 
    A clear explanation of the problem and why it matters
    A detailed description of your solution idea
    A simple system design or diagram (showing how your solution works)
    A simple demo
    A short plan explaining how your solution could be used in real life
 
Important: The expected output is a basic prototype, simulation, or model, and not simply presenting a concept or an idea.
Evaluation Criteria
Technical accuracy and modelling rigour
40%
 
 
How well the simulator reproduces historical hotel guest numbers
Creativity and originality
20%
 
How practical and realistic your idea is
20%
 
Clarity of your explanation and presentation
20%
