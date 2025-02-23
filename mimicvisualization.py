from google.colab import files
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

# Load patients data
d1 = files.upload()
patients = pd.read_csv('PATIENTS.csv')

# Load admissions data
d2 = files.upload()
admissions = pd.read_csv('ADMISSIONS.csv')

# Load ICUSTAYS data
d3 = files.upload()
icustays = pd.read_csv('ICUSTAYS.csv')

# Load PROCEDUREEVENTS_MV data
d4 = files.upload()
procedurevents = pd.read_csv('PROCEDUREEVENTS_MV.csv')

# Load D_ICD_PROCEDURES data
d5 = files.upload()
procedures_d_icd = pd.read_csv('D_ICD_PROCEDURES.csv')

# Load CAREGIVERS data
d6 = files.upload()
caregivers = pd.read_csv('CAREGIVERS.csv')

# Load DIAGNOSES_ICD data
d7 = files.upload()
diagnoses_icd = pd.read_csv('DIAGNOSES_ICD.csv')

# Load Prescriptions data
d8 = files.upload()
prescriptions = pd.read_csv('PRESCRIPTIONS.csv')

"""**1. Distribution of Drug Types**

This visualization uses a pie chart to show the distribution of drug types.


"""

import altair as alt

#Calculate drug type counts
drug_type_counts = prescriptions['drug_type'].value_counts().reset_index()
drug_type_counts.columns = ['Drug Type', 'Count']

#Calculate percentages
drug_type_counts['Percent'] = drug_type_counts['Count'] / drug_type_counts['Count'].sum()

#Base chart
base = alt.Chart(drug_type_counts).encode(
    theta=alt.Theta("Count", stack=True)
)

pie = base.mark_arc(outerRadius=120, innerRadius=80).encode(color=alt.Color("Drug Type"),
    order=alt.Order("Percent", sort="descending"),
    tooltip=["Drug Type", "Count", alt.Tooltip("Percent", format=".1%")])

text = base.mark_text(radius=140).encode(
    text=alt.Text("Percent", format=".1%"),
    order=alt.Order("Percent", sort="descending"),
    color=alt.value("black")
)

chart = (pie + text).properties(
    title='Distribution of Drug Types'
)
chart.show()

"""**2. Patient flow through a hospital system**

This visualization uses a Sankey diagram to show the flow of patients through a hospital system. The visualization shows the different admission types, ICU stays, and discharge locations.
"""

import plotly.graph_objects as go
import pandas as pd

#Merge the dataframes
merged_df = pd.merge(admissions, icustays, on=['subject_id', 'hadm_id'])
merged_df = pd.merge(merged_df, patients, on='subject_id')

#Define the nodes
nodes = [
    {'label': 'Emergency Admission', 'color': 'blue'},
    {'label': 'Elective Admission', 'color': 'green'},
    {'label': 'Urgent Admission', 'color': 'orange'},
    {'label': 'ICU Stay', 'color': 'red'},
    {'label': 'Discharge Home', 'color': 'purple'},
    {'label': 'Discharge SNF', 'color': 'pink'},
    {'label': 'Death', 'color': 'black'}
]

#Create dictionaries to map admission types and discharge locations to node indices
admission_type_to_node_index = {
    'EMERGENCY': 0,
    'ELECTIVE': 1,
    'URGENT': 2
}

discharge_location_to_node_index = {
    'HOME': 4,
    'SNF': 5,
    'DEAD/EXPIRED': 6,
    'DEAD/EXPIRED AT HOME': 6
}

links = []
for admission_type in merged_df['admission_type'].unique():
    source = admission_type_to_node_index[admission_type]
    target = 3
    value = len(merged_df[merged_df['admission_type'] == admission_type])
    links.append({'source': source, 'target': target, 'value': value})

for discharge_location in merged_df['discharge_location'].unique():
    source = 3
    target = discharge_location_to_node_index.get(discharge_location, 4)
    value = len(merged_df[merged_df['discharge_location'] == discharge_location])
    links.append({'source': source, 'target': target, 'value': value})

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=10,
        line=dict(color="black", width=0.5),
        label=[node['label'] for node in nodes],
        color=[node['color'] for node in nodes]
    ),
    link=dict(
        source=[link['source'] for link in links],
        target=[link['target'] for link in links],
        value=[link['value'] for link in links]
    )
)])

fig.update_layout(title_text="Patient Flow", font_size=10)
fig.show()

!pip install lifelines

"""**3. Kaplan-Meier survival curve**

This visualization uses a Kaplan-Meier survival curve to show the probability of survival over time.
"""

from lifelines import KaplanMeierFitter
import plotly.express as px

merged_df = pd.merge(admissions, patients, on='subject_id')

#Calculate survival time in years
merged_df['admittime'] = pd.to_datetime(merged_df['admittime'])
merged_df['dod'] = pd.to_datetime(merged_df['dod'])
merged_df['survival_time_years'] = (merged_df['dod'] - merged_df['admittime']).dt.days / 365.25

kmf = KaplanMeierFitter()

kmf.fit(merged_df['survival_time_years'], event_observed=merged_df['expire_flag'])

threshold = 0.5  # Set the threshold for survival probability

fig = px.line(
    x=kmf.survival_function_.index,
    y=kmf.survival_function_.values.flatten(),
    labels={'x': 'Time (years)', 'y': 'Survival Probability'},
    title='Kaplan-Meier Survival Curve'
)

#Add a horizontal line to indicate the threshold
fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text=f'Threshold ({threshold})')
fig.update_xaxes(tickangle=45)
fig.show()

"""**4. Caregiver Workload**

This visualization uses a bar chart and a side table to show the workload of caregivers. The bar chart shows the total workload for each caregiver, broken down by diagnosis. The side table shows the caregiver role, diagnosis, workload (in days), and percentage of total workload.
"""

import pandas as pd
import json

#Merge the 'PROCEDUREEVENTS_MV' and 'CAREGIVERS' tables on 'cgid'
merged_df = pd.merge(procedurevents, caregivers, on='cgid')

#Convert 'starttime' and 'endtime' to datetime
merged_df['starttime'] = pd.to_datetime(merged_df['starttime'])
merged_df['endtime'] = pd.to_datetime(merged_df['endtime'])

#Calculate the duration of each procedure
merged_df['duration'] = merged_df['endtime'] - merged_df['starttime']

#Merge 'merged_df' with 'df_admissions' on 'subject_id' and 'hadm_id'
merged_df = pd.merge(merged_df, admissions, on=['subject_id', 'hadm_id'])

#calculate the total workload
workload_df = merged_df.groupby(['cgid', 'label', 'diagnosis'])['duration'].sum().reset_index()

#Convert the duration column to the number of days
workload_df['duration_days'] = workload_df['duration'].dt.total_seconds() / 86400

#Calculate the total 'duration_days' and percentage
grouped_workload = workload_df.groupby(['label', 'diagnosis'])['duration_days'].sum().reset_index()
grouped_workload['percentage'] = grouped_workload['duration_days'] / grouped_workload['duration_days'].sum()

data = [{'label': row['label'], 'diagnosis': row['diagnosis'], 'workload': int(row['duration_days']), 'percentage': row['percentage']} for _, row in grouped_workload.iterrows()]

json_data = json.dumps(data)

#D3.js + HTML Code for Bar Chart with Header and Side Table
html_code = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Caregiver Workload</title>
<script src="https://d3js.org/d3.v6.min.js"></script>
<style>
    body {{
        font-family: Arial, sans-serif;
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 20px;
    }}
    h2 {{
        font-size: 24px;
        color: #333;
        margin-bottom: 20px;
    }}
  .container {{
        display: flex;
        gap: 30px;
        align-items: flex-start;
    }}
  .tooltip {{
        position: absolute;
        background-color: white;
        border: 1px solid #ccc;
        padding: 8px;
        border-radius: 4px;
        pointer-events: none;
        box-shadow: 0px 0px 5px rgba(0,0,0,0.3);
        font-size: 14px;
    }}
  .bar:hover {{
        opacity: 0.8;
        cursor: pointer;
    }}
    table {{
        border-collapse: collapse;
        width: 400px;  /* Increased width for diagnosis column */
        font-size: 14px;
    }}
    th, td {{
        border: 1px solid #ddd;
        padding: 8px;
        text-align: center;
    }}
    th {{
        background-color: #f2f2f2;
    }}
    tr:hover {{
        background-color: #f9f9f9;
        cursor: pointer;
    }}
</style>
</head>
<body>

<h2>Caregiver Workload</h2>

<div class="container">
    <div>
        <svg width="800" height="600"></svg>
        <div class="tooltip" style="opacity: 0;"></div>
    </div>

    <div>
        <h3>Caregiver Workload Summary</h3>
        <table id="data-table">
            <tr>
                <th>Caregiver Role</th>
                <th>Diagnosis</th>  <th>Workload (days)</th>
                <th>Percentage</th>
            </tr>
        </table>
    </div>
</div>

<script>
// Get the data
const data = {json_data}.sort((a, b) => d3.ascending(a.label, b.label));

const margin = {{top: 50, right: 50, bottom: 100, left: 80}},
    width = 800 - margin.left - margin.right,
    height = 600 - margin.top - margin.bottom;

const svg = d3.select("svg")
.attr("width", width + margin.left + margin.right)
.attr("height", height + margin.top + margin.bottom)
.append("g")
.attr("transform", `translate(${{margin.left}}, ${{margin.top}})`);

// Define color scale for diagnosis
const color = d3.scaleOrdinal(d3.schemeAccent)
.domain(data.map(d => d.diagnosis));

const x = d3.scaleBand()
.domain(data.map(d => d.label))
.range([0, width])
.padding(0.2);

const y = d3.scaleLinear()
.domain([0, d3.max(data, d => d.workload)])
.range([height, 0]);

svg.selectAll(".bar")
.data(data)
.enter()
.append("rect")
.attr("class", "bar")
.attr("x", d => x(d.label))
.attr("y", d => y(d.workload))
.attr("width", x.bandwidth())
.attr("height", d => height - y(d.workload))
.attr("fill", d => color(d.diagnosis))
.on("mouseover", function(event, d) {{
        d3.select(this).transition().duration(200).style("opacity", 0.7);
        tooltip.transition().duration(200).style("opacity", 0.9);
        tooltip.html(`<strong>${{d.label}}</strong><br>Diagnosis: ${{d.diagnosis}}<br>Workload: ${{d.workload}} days<br>Percentage: ${{d3.format(".1%")(d.percentage)}}`)
        .style("left", (event.pageX + 10) + "px")
        .style("top", (event.pageY - 28) + "px");
    }})
.on("mousemove", function(event) {{
        tooltip.style("left", (event.pageX + 10) + "px")
           .style("top", (event.pageY - 28) + "px");
    }})
.on("mouseout", function() {{
        d3.select(this).transition().duration(200).style("opacity", 1);
        tooltip.transition().duration(200).style("opacity", 0);
    }});

svg.append("g")
.attr("transform", `translate(0, ${{height}})`)
.call(d3.axisBottom(x))
.selectAll("text")
.attr("transform", "rotate(-45)")
.style("text-anchor", "end");

svg.append("g")
.call(d3.axisLeft(y));

svg.append("text")
.attr("x", (width / 2))
.attr("y", 0 - (margin.top / 2))
.attr("text-anchor", "middle")
.style("font-size", "16px")
.text("Caregiver Workload");

const tooltip = d3.select(".tooltip");

const table = d3.select("#data-table");
data.forEach((d, i) => {{
    const row = table.append("tr")
    .on("mouseover", function() {{
            d3.select(svg.selectAll(".bar").nodes()[i]).style("opacity", 0.8);
        }})
    .on("mouseout", function() {{
            d3.select(svg.selectAll(".bar").nodes()[i]).style("opacity", 1);
        }});

    row.append("td").text(d.label);
    row.append("td").text(d.diagnosis);  // Added Diagnosis cell
    row.append("td").text(d.workload);
    row.append("td").text(`${{d3.format(".1%")(d.percentage)}}`);
}});
</script>

</body>
</html>
"""

# Display the Bar Chart and Side Table
from IPython.display import display, HTML
display(HTML(html_code))

"""**5. Interactive Timeline of Patient Events**

This visualization uses an interactive timeline to display various patient events like admissions, discharges, ICU stays, and procedure start and end times
"""

#Convert time-related columns to datetime format
admissions['admittime'] = pd.to_datetime(admissions['admittime'])
admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
admissions['deathtime'] = pd.to_datetime(admissions['deathtime'])
icustays['intime'] = pd.to_datetime(icustays['intime'])
icustays['outtime'] = pd.to_datetime(icustays['outtime'])
procedurevents['starttime'] = pd.to_datetime(procedurevents['starttime'])
procedurevents['endtime'] = pd.to_datetime(procedurevents['endtime'])

events_df = pd.DataFrame()

#Admissions data
admissions_events = admissions[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime']].melt(id_vars=['subject_id', 'hadm_id'], value_name='event_time', var_name='event_type')
admissions_events['event_type'] = admissions_events['event_type'].replace({'admittime': 'Admission', 'dischtime': 'Discharge', 'deathtime': 'Death'})
events_df = pd.concat([events_df, admissions_events])

#ICU stays data
icustays_events = icustays[['subject_id', 'hadm_id', 'intime', 'outtime']].melt(id_vars=['subject_id', 'hadm_id'], value_name='event_time', var_name='event_type')
icustays_events['event_type'] = icustays_events['event_type'].replace({'intime': 'ICU Admission', 'outtime': 'ICU Discharge'})
events_df = pd.concat([events_df, icustays_events])

#Procedure events data
procedure_events = procedurevents[['subject_id', 'hadm_id', 'starttime', 'endtime']].melt(id_vars=['subject_id', 'hadm_id'], value_name='event_time', var_name='event_type')
procedure_events['event_type'] = procedure_events['event_type'].replace({'starttime': 'Procedure Start', 'endtime': 'Procedure End'})
events_df = pd.concat([events_df, procedure_events])

#Drop rows with null values in 'event_time' column of 'events_df'
events_df.dropna(subset = ['event_time'], inplace=True)

chart = alt.Chart(events_df).mark_point().encode(
    # Map 'event_time' to the x-axis
    x=alt.X('event_time', axis=alt.Axis(title='Event Time', format="%Y-%m-%d %H:%M:%S")),
    # Map 'subject_id' to the y-axis to show events for all subjects
    y=alt.Y('subject_id', axis=alt.Axis(title='Subject ID')),
    color='event_type',
    tooltip=['event_time', 'subject_id', 'hadm_id', 'event_type']
).properties(
    title='Timeline of Patient Events',
    width=1200,
    height=400,
).interactive()

chart.show()

#Count occurrences of each diagnosis
diagnosis_counts = admissions['diagnosis'].value_counts().head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=diagnosis_counts.index, y=diagnosis_counts.values)
plt.xlabel('Diagnosis')
plt.ylabel('Number of Patients')
plt.title('Top 10 Most Frequent Diagnoses')
plt.xticks(rotation=45, ha='right')
plt.show()