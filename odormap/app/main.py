import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import flask

import os
from os.path import join
import pandas as pd
import numpy as np
import pickle
import pyrfume

# Initialize app
external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css']
app = flask.Flask(__name__)
dapp = dash.Dash(__name__,
                server = app,
                url_base_pathname = '/',
                external_stylesheets=external_stylesheets)

# Paths and files
cwd = os.path.dirname(os.path.abspath(__file__))
path2data = join(cwd, 'data')

import pyrfume
import pandas as pd
file_path = pyrfume.DATA_DIR / 'odorants' / 'all_cids_properties.csv'
details = pd.read_csv(file_path, usecols=range(5), index_col=0)
file_path = pyrfume.DATA_DIR / 'odorants' / 'all_cids.csv'
sources = pd.read_csv(file_path, index_col=0)


###### BEGIN CONTENT ######

app.layout = html.Div(className='container', children=[
    html.Div(className='row', children=[
        dcc.Graph(
            id='basic-interactions',
            figure={
                'data': [
                    go.Scattergl(**{
                        'x': tsne_snitz.embedding_[:, 0],
                        'y': tsne_snitz.embedding_[:, 1],
                        'mode': 'markers',
                        'marker': {'size': 5,
                                   'color': clustering.labels_,
                                   'opacity': 0.5,
                                   'colorscale': 'rainbow'}
                    })],

                'layout': {
                    'clickmode': 'event+select',
                    'xaxis': {'type': 'linear', 
                              'title': '', 
                              'zeroline': False,
                              'showgrid': False,
                              'showline': False, 
                              'showticklabels': False},
                    'yaxis': {'type': 'linear',
                              'title': '',
                              'zeroline': False,
                              'showgrid': False,
                              'showline': False,
                              'showticklabels': False},
                    'margin': {'l': 40, 'b': 40, 't': 10, 'r': 10},
                    'legend': {'x': 0, 'y': 1},
                    'hovermode': 'closest',
                    'paper_bgcolor': 'rgba(0,0,0,0)',
                    'plot_bgcolor': 'rgba(0,0,0,0)',
                    'width': 750,
                    'height': 750,
                    },
                }
            ),]),

    html.Div(className='row', children=[
        html.Div(className='col', children=[
            html.Table([
                html.Tr([html.Td(['CID:']), 
                         html.Td([html.A(id='cid', 
                                         href='#')])]),
                html.Tr([html.Td(['MW:']), 
                         html.Td(id='mw')]),
                html.Tr([html.Td(['Name:']), 
                         html.Td(id='name')]),
                html.Tr([html.Td(['SMILES:']), 
                         html.Td(id='smiles')]),
                html.Tr([html.Td(['IUPAC:']), 
                         html.Td(id='iupac')]),
                html.Tr([html.Td(['Sources:']), 
                         html.Td(id='sources')]),
                ], style={'vertical-align': 'middle',
                          'font-size': '160%'})]),
        html.Div(className='col', children=[
            html.Img(id='molecule-2d', src='')
            ,]),
        ])
    ])


PC_1_props = ['AP2DelayMean','AP2DelayMeanStrongStim','AP1DelayMean']
PC_2_props = ['ISIMedian','SSAccommodationMean','InitialAccommodationMean']
PC_3_props = ['AP2RateOfChangePeakToTrough','AccommodationAtSSMean','AP1RateOfChangePeakToTrough']

# multiple callback error suppressed [Div has no .keys() atrribute]
dapp.config['suppress_callback_exceptions']=True



def dash_plot_ephys_clusters(display_prop_1,display_prop_2,display_prop_3,dash_container=0,
                             samples_df=cortical_ephys_df,raw_samples_df=cortical_ephys_no_trans_df,metadata_df=cortical_metadata_df):
    '''
        Plots clustering results for desired properties. Used in poster.

    '''

    display_props = [display_prop_1,display_prop_2,display_prop_3]


    # default values
    cluster_path = '/1/1/',
    cluster_captions = ['Fast Spikers (FS)',' Delayed Regular Spikers (dRS)','Bursters (B)',
                        'Non-Accommodating Regular Spikers (naRS)','Regular Spikers (RS)',
                        'Accommodating Fast Spikers (aFS)']


    # get desired clusters
    source_df = samples_df[samples_df["ClusterPath"].str.startswith(cluster_path)]
    clusters = source_df["Cluster"]
    source_df = raw_samples_df.loc[source_df.index]

    desired_models = source_df.index.tolist()

    if dash_container == 0:
        x_display = display_props[1]
        y_display = display_props[0]

        xs = source_df[x_display].values
        ys = source_df[y_display].values

        x_lim = [np.min(xs)-5,np.max(xs)+5]
        y_lim = [np.min(ys)-5,np.max(ys)+5]


    elif dash_container == 1:
        x_display = display_props[2]
        y_display = display_props[1]

        xs = source_df[x_display].values
        ys = source_df[y_display].values

        x_lim = [np.min(xs)-5,np.max(xs)+5]
        y_lim = [np.min(ys)-5,np.max(ys)+5]


    else:
        x_display = display_props[2]
        y_display = display_props[0]

        xs = source_df[x_display].values
        ys = source_df[y_display].values

        x_lim = [np.min(xs)-5,np.max(xs)+5]
        y_lim = [np.min(ys)-5,np.max(ys)+5]


    model_ids = source_df.index.tolist()
    reduced_metadata_df = metadata_df[metadata_df['Model_ID'].isin(model_ids)].sort_values(by=['Model_ID'])


    cluster_names = [cluster_captions[c_id] for c_id in clusters]
    pubs = reduced_metadata_df['Publication'].tolist()

    all_text = zip(model_ids,cluster_names,pubs)


    cluster_data = go.Scatter(
                x=xs, y=ys,
                text=['<b>%s</b><br><b>Cluster</b> : %s<br><b>Publication</b> : %s' %(mi,cn,p) for (mi,cn,p) in all_text],
                hovertemplate = "%{text}<br>"+
                                "<extra></extra><br>" +
                                "<i>%{yaxis.title.text}</i> : %{y:.02f}<br>" +
                                "<i>%{xaxis.title.text}</i> : %{x:.02f}<br>",
                mode='markers',
                marker={
                    'line' : {'width' : 1, 'color' : 'grey'},
                    'showscale' : False,
                    'colorscale' : 'Jet',
                    'color' : clusters,
                    'size' : 8,
                    'opacity' : 0.6})


    figure = {
        'data' : [cluster_data],
        'layout' : go.Layout(title=x_display + ' vs ' + y_display,
                              showlegend=False, hovermode='closest',
                              margin={'b': 20, 'l': 5, 'r': 5, 't': 40},
                              xaxis={'title' : x_display,
                                    'range' : x_lim, 'automargin' : True,
                                    'showgrid': True, 'gridwidth' : 2,
                                    'zeroline': False, 'showticklabels': True},
                              yaxis={'title' : y_display, 'automargin' : True,
                                    'range' : y_lim,
                                    'showgrid': True, 'gridwidth' : 2,
                                    'zeroline': False, 'showticklabels': True}
                              )
                }

    return figure




dapp.layout = html.Div([

        html.Div([html.H1("Automated Assessment and Comparison of Cortical Neuron Models")

        ],style={"textAlign" : "center"}),

        html.Div([
            dcc.Graph(
                    id='cluster-graph_1',
                    figure = dash_plot_ephys_clusters(samples_df=cortical_ephys_df,
                                                      raw_samples_df=cortical_ephys_no_trans_df,
                                                      metadata_df=cortical_metadata_df,
                                                      display_prop_1=PC_1_props[2],
                                                      display_prop_2=PC_2_props[0],
                                                      display_prop_3=PC_3_props[1],
                                                      dash_container=0)

                )
        ], className="container_1",
           style = {"height" : "33%", "width" : "33%",  'display': 'inline-block'}),


         html.Div([
             dcc.Graph(
                     id='cluster-graph_2',
                     figure = dash_plot_ephys_clusters(samples_df=cortical_ephys_df,
                                                      raw_samples_df=cortical_ephys_no_trans_df,
                                                      metadata_df=cortical_metadata_df,
                                                      display_prop_1=PC_1_props[2],
                                                      display_prop_2=PC_2_props[0],
                                                      display_prop_3=PC_3_props[1],
                                                      dash_container=1)

                 )
         ], className="container_2",
            style = {"height" : "33%", "width" : "33%",  'display': 'inline-block'}),


        html.Div([
            dcc.Graph(
                    id='cluster-graph_3',
                    figure = dash_plot_ephys_clusters(samples_df=cortical_ephys_df,
                                                     raw_samples_df=cortical_ephys_no_trans_df,
                                                     metadata_df=cortical_metadata_df,
                                                     display_prop_1=PC_1_props[2],
                                                     display_prop_2=PC_2_props[0],
                                                     display_prop_3=PC_3_props[1],
                                                     dash_container=2)

                )
        ], className="container_3",
           style = {"height" : "33%", "width" : "33%",  'display': 'inline-block'}),


          # START USER INPUT FILTER
          html.Div([html.H6("Filter By Features", className="row",
                        style={"display": "inline-block","margin-left": "37%","text-align": "center", "text-decoration": "underline", 'width':'25%'}),
                        ]),

          html.Div([

                  # USER INPUT UPDATES
                  html.Label(id='feature-1',children=[dcc.Dropdown(id='pc-1-input', options=[{"label": i, "value": i} for i in PC_1_props],clearable=False,
                                               value=PC_1_props[2], className="component 1")],style ={'display' : 'inline-block','width':'33%'}),

                  html.Label(id='feature-2',children=[dcc.Dropdown(id='pc-2-input', options=[{"label": i, "value": i} for i in PC_2_props],clearable=False,
                                                         value=PC_2_props[0], className="component 2")],style ={'display' : 'inline-block','width':'33%'}),

                  html.Label(id='feature-3',children=[dcc.Dropdown(id='pc-3-input', options=[{"label": i, "value": i} for i in PC_3_props],clearable=False,
                                                         value=PC_3_props[1], className="component 3")],style ={'display' : 'inline-block', 'width':'33%'})

                         ], style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "100%"}),

          # RESET USER INPUTS
          html.Button('Reset Features',id='reset-input', n_clicks=0,
                  style={'display':'block',"margin-left": "37%",'text-align':'center','width':'25%'}),
  ])


@dapp.callback(
    [Output("pc-1-input","value"),Output("pc-2-input","value"),Output("pc-3-input","value")],
    [Input('reset-input','n_clicks')],
    [State('pc-1-input','value'),State('pc-2-input','value'),State('pc-3-input','value')])
def update_clicks(clicks,display_props_1,display_props_2,display_props_3):
    if clicks>0:
        return PC_1_props[2], PC_2_props[0], PC_3_props[1]
    else:
        return display_props_1, display_props_2, display_props_3



@dapp.callback(
    Output("cluster-graph_1","figure"),
    [Input("pc-1-input", "value"), Input("pc-2-input", 'value'), Input("pc-3-input", 'value')])
def update_figure_1(display_prop_1,display_prop_2,display_prop_3):

    figure = dash_plot_ephys_clusters(display_prop_1, display_prop_2,display_prop_3,
                                      samples_df=cortical_ephys_df,
                                      raw_samples_df=cortical_ephys_no_trans_df,
                                      dash_container=0)


    return figure

@dapp.callback(
    Output("cluster-graph_2","figure"),
    [Input("pc-1-input", "value"), Input("pc-2-input", 'value'), Input("pc-3-input", 'value')])
def update_figure_1(display_prop_1,display_prop_2,display_prop_3):

    figure = dash_plot_ephys_clusters(display_prop_1, display_prop_2,display_prop_3,
                                      samples_df=cortical_ephys_df,
                                      raw_samples_df=cortical_ephys_no_trans_df,
                                      dash_container=1)


    return figure

@dapp.callback(
    Output("cluster-graph_3","figure"),
    [Input("pc-1-input", "value"), Input("pc-2-input", 'value'), Input("pc-3-input", 'value')])
def update_figure_1(display_prop_1,display_prop_2,display_prop_3):

    figure = dash_plot_ephys_clusters(display_prop_1, display_prop_2,display_prop_3,
                                      samples_df=cortical_ephys_df,
                                      raw_samples_df=cortical_ephys_no_trans_df,
                                      dash_container=2)


    return figure

###### END CONTENT ######




if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=80)
