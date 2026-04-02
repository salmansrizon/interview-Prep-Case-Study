![Tux, the Linux mascot](cover-photo.png)

### Introduction
Welcome to your first day as the Lead Data Analyst for the central division of a major urban mobility platform.

You haven't even finished your morning coffee when the Regional General Manager calls an emergency meeting. The executive team is panicking, department heads are pointing fingers, and no one knows exactly what is going wrong with the platform's core operations. They need answers, and they need them today.

You will not just be writing SQL syntax; you will be acting as the central intelligence for a multi-million dollar company. Your objective is to query raw, real-world database records to map revenue streams, diagnose operational failures, and provide actionable insights to save the marketplace.

### Business & Industry Context

To solve this problem, you must first understand the battlefield.

The Industry Reality
The ride-sharing industry operates as a Two-Sided Marketplace. The company does not own the cars, nor does it employ the riders. The entire business model relies on the technology that connects Supply (Drivers) with Demand (Riders).

**Razor-Thin Margins:** The platform makes pennies on the dollar per ride. Profitability relies entirely on massive scale and extreme operational efficiency.

**Zero Switching Costs:** Loyalty is practically non-existent. If a rider opens the app and sees a 15-minute wait time, it takes them exactly 3 seconds to open a competitor's app.

**Volatile Supply:** Drivers are independent contractors. If they feel they aren't making enough money, or if they are forced to drive long distances for cheap fares, they simply turn off the app.

The Company's Current State
You operate in a dense, highly populated metropolitan area. The platform offers multiple tiers of service—from cheap 2-wheeler 'Autos' to luxury 'Premium SUVs'—to capture every type of user.

> The Good News is  Marketing has done an incredible job. Brand awareness is at an all-time high. In the latest quarterly snapshot, the app generated nearly 150,000 ride requests.

> The Bad Newsis Top-line requests mean nothing if the platform cannot fulfill them. Operations just ran a preliminary report revealing a massive operational failure.


### The Problem

#### The Leaky Funnel 

Out of the 150,000 ride requests generated this quarter, only 66% resulted in a completed, paid trip. One out of every three customers who opens the app, types in a destination, and actively wants to pay for a service is failing to do so. The company is bleeding millions of dollars in potential revenue and heavily damaging its reputation.

Internal departments are deadlocked over the root cause:

- The Product Team suspects the app's routing algorithm is broken. They believe estimated wait times (avg_vtat) are too high, causing riders to get frustrated and cancel.

- The Operations Team blames the fleet. They hypothesize that drivers are cherry-picking expensive trips and actively rejecting lower-tier users.

- The Finance Team believes there is a payment friction issue, suspecting that rides booked with 'Cash' are inherently less reliable than digital prepayments.


#### The Rating vs. Revenue Paradox

The Quality Assurance (QA) team assumes that highly rated drivers generate more revenue because they provide a better experience. However, the Operations team thinks lower-rated drivers might actually generate more revenue because they hustle and take every ride, regardless of distance. We need to prove who is right.

- Segment completed rides into driver rating buckets (e.g., 'Excellent', 'Average', 'Poor') and calculate the average booking value for each bucket.


#### The "True Typical" Ride (Percentiles vs. Averages)

The Finance team is using the average booking value to forecast next quarter's revenue. However, a few extremely long cross-country rides are skewing the average way up. They need to know the median booking value for each vehicle type to understand what a truly "typical" ride looks like.

- Calculate both the Average and the Median (50th percentile) booking value for completed rides, grouped by vehicle type.


#### Identifying "Whale" Routes (Platform Contribution Percentage)

The Marketing team has a limited budget to run outdoor billboard ads. They only want to place billboards on the specific A-to-B routes that drive the platform's revenue.

- Rank the top 10 most lucrative pickup-to-dropoff routes, and calculate exactly what percentage of the entire platform's total revenue each specific route contributes.

### Deliverables

Your mission is to cut through the internal noise using the raw ride_bookings database. You will write SQL queries to diagnose the bottleneck across three core pillars.


## Data Overview:

The dataset captures 148,770 total bookings across multiple vehicle types and provides a complete view of ride-sharing operations including successful rides, cancellations, customer behaviors, and financial metrics.

Key Statistics:
- Total Bookings: 148.77K rides
- Success Rate: 65.96% (93K completed rides)
- Cancellation Rate: 25% (37.43K cancelled bookings)
- Customer Cancellations: 19.15% (27K rides)
- Driver Cancellations: 7.45% (10.5K rides)

### Data Schema

You have been granted access to the raw operations table. Here are the core columns you will use to run your diagnostics:

| Column Name | Description |
|---|---|
| Date | Date of the booking |
| Time | Time of the booking |
| Booking ID | Unique identifier for each ride booking |
| Booking Status | Status of booking (Completed, Cancelled by Customer, Cancelled by Driver, etc.) |
| Customer ID | Unique identifier for customers |
| Vehicle Type | Type of vehicle (Go Mini, Go Sedan, Auto, eBike/Bike, UberXL, Premier Sedan) |
| Pickup Location | Starting location of the ride |
| Drop Location | Destination location of the ride |
| Avg VTAT | Average time for driver to reach pickup location (in minutes) |
| Avg CTAT | Average trip duration from pickup to destination (in minutes) |
| Cancelled Rides by Customer | Customer-initiated cancellation flag |
| Reason for cancelling by Customer | Reason for customer cancellation |
| Cancelled Rides by Driver | Driver-initiated cancellation flag |
| Driver Cancellation Reason | Reason for driver cancellation |
| Incomplete Rides | Incomplete ride flag |
| Incomplete Rides Reason | Reason for incomplete rides |
| Booking Value | Total fare amount for the ride |
| Ride Distance | Distance covered during the ride (in km) |
| Driver Ratings | Rating given to driver (1-5 scale) |
| Customer Rating | Rating given by customer (1-5 scale) |
| Payment Method | Method used for payment (UPI, Cash, Credit Card, Uber Wallet, Debit Card) |



Revenue Distribution by Payment Method
- UPI: Highest contributor (~40% of total revenue)
- Cash: Second highest (~25% of total revenue)
- Credit Card: ~15% of total revenue
- Uber Wallet: ~12% of total revenue
- Debit Card: ~8% of total revenue

Cancellation Patterns
- Customer Cancellation Reasons:
- Wrong Address: 22.5%
- Driver Issues: 22.4%
- Driver Not Moving: 22.2%
- Change of Plans: 21.9%
- App Issues: 11.0%

Driver Cancellation Reasons:
- Capacity Issues: 25.0%
- Customer Related Issues: 25.3%
- Personal & Car Issues: 24.9%
- Customer Behavior: 24.8%

Rating Analysis
- Customer Ratings: Consistently high across all vehicle types (4.40-4.41)
- Driver Ratings: Slightly lower but stable (4.23-4.24)
- Highest Rated: Go Sedan (4.41 customer rating)
- Most Satisfied Drivers: UberXL category (4.24 rating)

Data Quality
- Completeness: Comprehensive coverage with minimal missing values
- Consistency: Standardized vehicle types and status categories
- Temporal Coverage: Full year 2024 data with daily granularity
- Geographic Scope: Multiple pickup and drop locations
- Balanced Distribution: Good representation across all vehicle types and time periods.

you can also see the [dataset](https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard/data?select=ncr_ride_bookings.csv).


### Considarations

You must write queries to solve specific business questions for the three warring departments


Finance & Profitability
- Determine which payment methods drive actual revenue (Completed rides) versus perceived revenue.

- Identify if premium vehicle tiers are generating enough consistent revenue to justify their maintenance.

Product & UX (The Cancellation Autopsy)

- Separate rider cancellations from driver cancellations to calculate specific cancellation rates per vehicle type.

- Group wait times into logical buckets to prove or disprove the theory that high wait times cause rider churn.

Operations & Hyper-Local Efficiency

- Identify the top 5 geographical "danger zones" (pickup locations) suffering from the highest ride volumes but the lowest completion yields.

- Calculate the average driver rating in those specific danger zones to check for quality control issues.


Need to create a performance dashboard base on the requirement