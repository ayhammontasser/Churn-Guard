# ------------------------------------------------------
# Telco Customer Churn Dashboard (Dash + Plotly + Pandas)
# Pages: Overview | Customer Segmentation | Churn Analysis | CLV
# ------------------------------------------------------

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import numpy as np

# ------------------------------------------------------
# 1. LOAD AND PREPARE DATA
# ------------------------------------------------------

def load_and_prepare():
    df = pd.read_csv("Telco_Final_Dataset.csv")

    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    
    df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'])
    df['AvgCharges'] = df.get('AvgCharges', df['MonthlyCharges'])

 
    df['tenure_years'] = (df['tenure'] / 12).round(2)


    df['NumServices'] = df[[
        'PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
        'DeviceProtection','TechSupport','StreamingTV','StreamingMovies'
    ]].apply(lambda row: sum([
        1 if (isinstance(v, str) and v.lower() in ['yes', 'yes internet'])
        or (not isinstance(v, str) and bool(v)) else 0 for v in row
    ]), axis=1)

    # CLV (Customer Lifetime Value)
    if 'CLV' not in df.columns:
        df['CLV'] = (df['MonthlyCharges'] * df['tenure']).fillna(0)

    # 
    if 'RiskSegment' not in df.columns:
        df['risk_score'] = (df['MonthlyCharges'] / (df['MonthlyCharges'].max()+1)) * \
                           (1 - (df['tenure'] / (df['tenure'].max()+1)))
        df['RiskSegment'] = pd.qcut(df['risk_score'].rank(method='first'), 3, labels=['Low','Medium','High'])

    # churn values
    df['Churn'] = df['Churn'].astype(str).str.strip().replace({'1':'Yes','0':'No'}).replace({1:'Yes',0:'No'})

    # LoyaltyClass & RevenueSegment
    if 'LoyaltyClass' not in df.columns:
        df['LoyaltyClass'] = pd.cut(df['tenure'], bins=[-1,6,24,999], labels=['New','Mid','Loyal'])
    if 'RevenueSegment' not in df.columns:
        df['RevenueSegment'] = pd.qcut(df['MonthlyCharges'].rank(method='first'), 3, labels=['Low','Medium','High'])

    # CLV Segmentation
    df['CLVSegment'] = pd.qcut(df['CLV'].rank(method='first'), 3, labels=['Low','Medium','High'])

    return df

df = load_and_prepare()

# ------------------------------------------------------
# 2. INITIALIZE DASH APP
# ------------------------------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
server = app.server

# ------------------------------------------------------
# 3. LAYOUT COMPONENTS
# ------------------------------------------------------

def kpi_card(title, value, subtitle=""):
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="card-title"),
            html.H3(value, className="card-text"),
            html.Small(subtitle, className="text-muted")
        ]),
        className="m-2 p-2 shadow-sm"
    )

# Sidebar filters
sidebar = dbc.Card(
    [
        html.H5("Filters", className="mt-2"),
        html.Label("Gender"),
        dcc.Dropdown(id='filter-gender',
                     options=[{'label': g, 'value': g} for g in sorted(df['gender'].dropna().unique())],
                     value=None, placeholder="All", clearable=True),
        html.Br(),
        html.Label("Contract"),
        dcc.Dropdown(id='filter-contract',
                     options=[{'label': c, 'value': c} for c in sorted(df['Contract'].dropna().unique())],
                     value=None, placeholder="All", clearable=True),
        html.Br(),
        html.Label("Internet Service"),
        dcc.Dropdown(id='filter-internet',
                     options=[{'label': s, 'value': s} for s in sorted(df['InternetService'].dropna().unique())],
                     value=None, placeholder="All", clearable=True),
        html.Br(),
        html.Label("Risk Segment"),
        dcc.Checklist(id='filter-risk',
                      options=[{'label': r, 'value': r} for r in sorted(df['RiskSegment'].unique())],
                      value=list(df['RiskSegment'].unique()), inline=True),
        html.Hr(),
        dbc.Button("Apply Filters", id='apply-filters', color='primary', className='mb-3'),
    ],
    body=True
)

# Navigation tabs
tabs = dbc.Tabs(
    [
        dbc.Tab(label="Overview", tab_id="overview"),
        dbc.Tab(label="Customer Segmentation", tab_id="segmentation"),
        dbc.Tab(label="Churn Analysis", tab_id="churn_analysis"),
        dbc.Tab(label="CLV", tab_id="clv"),
    ],
    id="tabs",
    active_tab="overview",
)

page_content = html.Div(id='page-content', className='p-3')

app.layout = dbc.Container([
    html.H1("Customer Churn Prediction Dashboard", className='text-center my-3 text-primary'),
    dbc.Row([
        dbc.Col(sidebar, width=3),
        dbc.Col([
            tabs,
            html.Div(id='kpi-row', className='d-flex flex-wrap'),
            html.Hr(),
            page_content
        ], width=9)
    ])
], fluid=True)

# ------------------------------------------------------
# 4. HELPER: FILTER FUNCTION
# ------------------------------------------------------

def apply_filters(dff, gender, contract, internet, risk_values):
    if gender:
        dff = dff[dff['gender']==gender]
    if contract:
        dff = dff[dff['Contract']==contract]
    if internet:
        dff = dff[dff['InternetService']==internet]
    if risk_values:
        dff = dff[dff['RiskSegment'].isin(risk_values)]
    return dff

# ------------------------------------------------------
# 5. CALLBACK TO UPDATE KPI + PAGE
# ------------------------------------------------------

@app.callback(
    Output('kpi-row', 'children'),
    Output('page-content', 'children'),
    Input('apply-filters', 'n_clicks'),   # 1st Input
    Input('tabs', 'active_tab'),          # 2nd Input (IMPORTANT: second input)
    State('filter-gender', 'value'),
    State('filter-contract', 'value'),
    State('filter-internet', 'value'),
    State('filter-risk', 'value'),
    prevent_initial_call=False
)
def update_main(n_clicks, active_tab, gender, contract, internet, risk_values):
    """
    NOTE:
    - The order of function arguments MUST match the order of Inputs then States in the decorator.
      Here: n_clicks (Input1), active_tab (Input2), then gender/contract/internet/risk_values (States).
    """

    # apply filters (uses the State values)
    dff = apply_filters(df, gender, contract, internet, risk_values)

    # KPIs
    total_customers = len(dff)
    total_churn = dff['Churn'].str.lower().eq('yes').sum()
    total_revenue = dff['TotalCharges'].sum() if 'TotalCharges' in dff.columns else 0
    years = round(dff['tenure'].max() / 12, 1) if not dff['tenure'].isnull().all() else 0

    kpis = dbc.Row([
        dbc.Col(kpi_card("Churn Count", f"{total_churn}", "Number of churned customers"), width=3),
        dbc.Col(kpi_card("Total Customers", f"{total_customers}", ""), width=3),
        dbc.Col(kpi_card("Years (approx.)", f"{years}", ""), width=3),
        dbc.Col(kpi_card("Total Revenue", f"${total_revenue:,.0f}", ""), width=3),
    ], className='mb-3')

    # choose page content based on active_tab
    if active_tab == 'overview':
        content = overview_page(dff)
    elif active_tab == 'segmentation':
        content = segmentation_page(dff)
    elif active_tab == 'churn_analysis':
        content = churn_analysis_page(dff)
    elif active_tab == 'clv':
        content = clv_page(dff)
    else:
        content = html.Div("Select a page")

    return kpis, content


# ------------------------------------------------------
# 6. PAGE FUNCTIONS
# ------------------------------------------------------

def overview_page(dff):
    churn_by_contract = dff.groupby(['Contract','Churn']).size().reset_index(name='count')
    bar_fig = px.bar(churn_by_contract, x='Contract', y='count', color='Churn', barmode='group', title='Churn by Contract Type')

    # FIXED: Gender pie chart
    gender_count = dff['gender'].value_counts().reset_index()
    gender_count.columns = ['gender', 'count']
    gender_pie = px.pie(gender_count, names='gender', values='count', hole=0.5, title='Customer Count by Gender')

    # Line: Revenue vs Tenure
    revenue_tenure = dff.groupby('tenure_years', as_index=False)['TotalCharges'].sum()
    line_fig = px.line(revenue_tenure, x='tenure_years', y='TotalCharges', markers=True, title='Revenue vs Tenure (years)')

    # FIXED: Internet Service Distribution
    internet_count = dff['InternetService'].value_counts().reset_index()
    internet_count.columns = ['InternetService', 'count']
    internet_pie = px.pie(internet_count, names='InternetService', values='count', hole=0.5, title='Internet Service Distribution')

    layout = dbc.Container([
        dbc.Row([dbc.Col(dcc.Graph(figure=bar_fig), width=6), dbc.Col(dcc.Graph(figure=gender_pie), width=6)]),
        dbc.Row([dbc.Col(dcc.Graph(figure=line_fig), width=6), dbc.Col(dcc.Graph(figure=internet_pie), width=6)]),
    ], fluid=True)
    return layout


def segmentation_page(dff):
    seg_df = dff.groupby(['RiskSegment','Churn']).size().reset_index(name='count')
    seg_bar = px.bar(seg_df, x='RiskSegment', y='count', color='Churn', barmode='group', title='Customers by Risk Segment and Churn')

    rev_df = dff.groupby('RevenueSegment', as_index=False)['TotalCharges'].sum()
    rev_pie = px.pie(rev_df, names='RevenueSegment', values='TotalCharges', hole=0.5, title='Total Charges by Revenue Segment')

    tenure_df = dff.groupby('LoyaltyClass', as_index=False)['tenure_years'].median()
    tenure_bar = px.bar(tenure_df, x='LoyaltyClass', y='tenure_years', title='Median Tenure (years) by Loyalty Class')

    loy_df = dff.groupby(['LoyaltyClass','Churn']).size().reset_index(name='count')
    loy_bar = px.bar(loy_df, x='LoyaltyClass', y='count', color='Churn', barmode='group', title='Customers by Loyalty Class and Churn')

    layout = dbc.Container([
        dbc.Row([dbc.Col(dcc.Graph(figure=seg_bar), width=6), dbc.Col(dcc.Graph(figure=rev_pie), width=6)]),
        dbc.Row([dbc.Col(dcc.Graph(figure=tenure_bar), width=6), dbc.Col(dcc.Graph(figure=loy_bar), width=6)]),
    ], fluid=True)
    return layout


def churn_analysis_page(dff):
    churned = dff[dff['Churn'].str.lower()=='yes']
    churn_count = len(churned)
    sum_tenure = churned['tenure'].sum() if not churned.empty else 0
    sum_total_charges = churned['TotalCharges'].sum() if not churned.empty else 0

    persona_bar = px.bar(churned.groupby('Partner', as_index=False).size(), x='Partner', y='size', title='Churned by Partner Status')

    churn_by_internet = churned['InternetService'].value_counts().reset_index()
    churn_by_internet.columns = ['InternetService', 'count']
    churn_internet_pie = px.pie(churn_by_internet, names='InternetService', values='count', title='Churn by Internet Service')

    churn_by_payment = churned['PaymentMethod'].value_counts().reset_index()
    churn_by_payment.columns = ['PaymentMethod', 'count']
    churn_payment_pie = px.pie(churn_by_payment, names='PaymentMethod', values='count', title='Churn by Payment Method')

    service_cols = ['DeviceProtection','StreamingMovies','OnlineBackup','StreamingTV','OnlineSecurity','TechSupport']
    service_counts = [(col, churned[col].str.lower().eq('yes').sum()) for col in service_cols if col in churned.columns]
    serv_df = pd.DataFrame(service_counts, columns=['service','count']).sort_values('count', ascending=True)
    service_bar = px.bar(serv_df, x='count', y='service', orientation='h', title='Services used by churned customers')

    layout = dbc.Container([
        dbc.Row([
            dbc.Col(kpi_card("Churn Count", f"{churn_count}"), width=3),
            dbc.Col(kpi_card("Sum Tenure (months)", f"{sum_tenure}"), width=3),
            dbc.Col(kpi_card("Sum Total Charges (churned)", f"${sum_total_charges:,.0f}"), width=3),
        ], className='mb-3'),
        dbc.Row([dbc.Col(dcc.Graph(figure=persona_bar), width=6), dbc.Col(dcc.Graph(figure=churn_internet_pie), width=6)]),
        dbc.Row([dbc.Col(dcc.Graph(figure=churn_payment_pie), width=6), dbc.Col(dcc.Graph(figure=service_bar), width=6)])
    ], fluid=True)
    return layout


def clv_page(dff):
    avg_clv = dff['CLV'].mean()
    sum_avgcharges = dff['AvgCharges'].sum() if 'AvgCharges' in dff.columns else dff['MonthlyCharges'].sum()
    tenure_count = dff['tenure'].sum()

    scatter = px.scatter(dff, x='tenure_years', y='CLV', color='LoyaltyClass', hover_data=['customerID'], title='CLV vs Tenure by Loyalty Class')
    clv_bar = px.bar(dff.groupby('LoyaltyClass', as_index=False)['CLV'].mean(), x='LoyaltyClass', y='CLV', title='Average CLV by Loyalty Class')

    clv_tenure = dff.groupby('tenure_years', as_index=False)['CLV'].sum()
    line_clv_tenure = px.line(clv_tenure, x='tenure_years', y='CLV', title='Sum of CLV by Tenure (years)')

    clv_risk = dff.groupby('RiskSegment', as_index=False)['CLV'].sum()
    line_clv_risk = px.bar(clv_risk, x='RiskSegment', y='CLV', title='Total CLV by Risk Segment')

    layout = dbc.Container([
        dbc.Row([
            dbc.Col(kpi_card("Avg CLV", f"${avg_clv:,.0f}"), width=3),
            dbc.Col(kpi_card("Sum AvgCharges", f"${sum_avgcharges:,.0f}"), width=3),
            dbc.Col(kpi_card("Sum Tenure", f"{tenure_count}", ""), width=3),
        ], className='mb-3'),
        dbc.Row([dbc.Col(dcc.Graph(figure=scatter), width=6), dbc.Col(dcc.Graph(figure=clv_bar), width=6)]),
        dbc.Row([dbc.Col(dcc.Graph(figure=line_clv_tenure), width=6), dbc.Col(dcc.Graph(figure=line_clv_risk), width=6)])
    ], fluid=True)
    return layout

# ------------------------------------------------------
# 7. RUN APP
# ------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=8050)
