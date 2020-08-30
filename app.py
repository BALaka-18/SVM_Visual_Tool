import os,time,dash
import pandas as pd
import numpy as np
from textwrap import dedent
# Dash and Plotly
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input,Output,State
import plotly.graph_objs as go
# Sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,roc_auc_score

# Backend to generate dataset
def gen_dat(n_samples,noise,dataset_type):
    if dataset_type == 'non_lin1':
        return datasets.make_circles(n_samples=n_samples,noise=noise,factor=0.5,random_state=1)
    elif dataset_type == 'non_lin2':
        return datasets.make_moons(n_samples=n_samples,noise=noise,random_state=0)
    elif dataset_type == 'lin':
        X,Y = datasets.make_classification(n_samples=n_samples,n_features=2,n_redundant=0,n_informative=2,random_state=2,n_clusters_per_class=1)
        # Add noise to features
        range_noise = np.random.RandomState(seed=2)
        X += noise*(range_noise.uniform(size=X.shape))
        lin_data = (X,Y)
        return lin_data
    else:
        raise ValueError("Select correct dataset type !")
# Backend for SVM plot
def plot_svm(model,trainx,testx,trainy,testy,Z,XX,YY,mesh_step,threshold):
    # Get metrics
    y_pred_train = (model.decision_function(trainx) > threshold).astype(int)   
    # Segregating : Those greater than threshold, lie above threshold line and are assigned to one class. Vice-versa for those below threshold. Converting bool to int.
    y_pred_test = (model.decision_function(testx) > threshold).astype(int)
    train_score = accuracy_score(y_true=trainy,y_pred=y_pred_train)
    test_score = accuracy_score(y_true=testy,y_pred=y_pred_test)

    #Computing threshold(Xing Han)(manually scaling the threshold)
    thres_scaled = threshold*(Z.max() - Z.min()) + Z.min()
    range_thres = max(abs(thres_scaled - Z.min()),abs(thres_scaled - Z.max()))

    # Plotting
    # Trace 0 : SVM contour plot
    trace0 = go.Contour(
        x=np.arange(XX.min(),XX.max(),mesh_step),
        y=np.arange(YY.min(),YY.max(),mesh_step),
        z=Z.reshape(XX.shape),
        zmin=float(thres_scaled - range_thres),
        zmax=float(thres_scaled + range_thres),
        showscale=False,
        contours=dict(showlines=False),
        opacity=0.6
    )
    # Trace 1 : Threshold line
    trace1 = go.Contour(
        x=np.arange(XX.min(),XX.max(),mesh_step),
        y=np.arange(YY.min(),YY.max(),mesh_step),
        z=Z.reshape(XX.shape),
        showscale=False,
        contours=dict(
            showlines=False,
            type='constraint',
            operation='=',
            value=thres_scaled,
        ),
        # Display updated scaled threshold value
        name=f'Threshold value = {thres_scaled:.3f}',
        line=dict(color='#000000')
    )
    # Trace 2 : Training data
    trace2 = go.Scatter(
        x=trainx[:,0],
        y=trainx[:,1],
        mode='markers',
        name=f'Training Data (accuracy={train_score:.3f})',
        marker=dict(
            size=12,
            color=trainy,
            line=dict(width=1)
        )
    )
    # Trace 3 : Test date
    trace3 = go.Scatter(
        x=testx[:,0],
        y=testx[:,1],
        mode='markers',
        name=f'Test Data (accuracy={test_score:.3f})',
        marker=dict(
            size=12,
            symbol='triangle-up',
            color=testy,
            line=dict(width=1)
        )
    )
    # Putting all together
    layout = go.Layout(
        xaxis=dict(ticks='',showticklabels=False,showgrid=False,zeroline=False,),
        yaxis=dict(ticks='',showticklabels=False,showgrid=False,zeroline=False,),
        hovermode='closest',
        legend=dict(x=0,y=-0.01,orientation='h'),
        margin=dict(l=0,r=0,t=0,b=0)  # Left,Right,Top,Bottom
    )
    data = [trace0,trace1,trace2,trace3]
    figure = go.Figure(data=data,layout=layout)

    return figure

# Backend for confusion matrix pie chart
def pie_conf(model,testx,testy,Z,threshold):
    # Re-computing scaled threshold
    thres_scaled = threshold*(Z.max() - Z.min()) + Z.min()
    y_pred_test = (model.decision_function(testx) > thres_scaled).astype(int)

    # Computing confusion matrix and related metrics
    matrix = confusion_matrix(y_true=testy,y_pred=y_pred_test)
    tn, fp, fn, tp = matrix.ravel()
    values = [tp, fn, fp, tn]
    labels_ident = [
        'True Positive',
        'False Negative',
        'False Positive',
        'True Negative'
    ]
    labels = ['TP','FN','FP','TN']
    # Plot the pie chart
    trace0 = go.Pie(
        labels=labels_ident,
        values=values,
        hoverinfo='label+value+percent',
        textinfo='text+value',
        text=labels,
        sort=False
    )
    # Putting all together
    layout = go.Layout(
        title=f'Confusion Matrix Pie Chart',
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(bgcolor='rgba(255,255,255,0)',orientation='h')
    )
    data = [trace0]
    figure = go.Figure(data=data,layout=layout)

    return figure

# Backend for 3D plot of Gaussian rbf kernel
def plot_gaus3d(model,X,Y,type='gaussian'):
    # Defaulting gamma to 0.3
    projected = np.exp(-(X**2).sum(1)*0.3)
    # For wireframe plot
    X_new = np.insert(X,2,projected,axis=1)
    new_model = SVC(
        kernel='linear',
        C=1.0
    )
    clf = new_model.fit(X_new,Y)
    Z = lambda X,Y: (-clf.intercept_[0]-clf.coef_[0][0]*X-clf.coef_[0][1]*Y) / clf.coef_[0][2]
    # 3D plot
    trace0 = go.Mesh3d(
        x = X[:,0], 
        y = X[:,1], 
        z = Z(X[:,0],X[:,1])
    ) ## for separating plane
    trace1 = go.Scatter3d(
        x=X[:,0],
        y=X[:,1],
        z=projected,
        mode='markers',
        marker=dict(size = 3,color = Y,colorscale = 'Viridis')
    )
    layout = go.Layout(
        title=f'3D Demonstration(double tap and hover to move the graph)',
        margin=dict(l=10, r=10, t=60, b=10)
    )
    data = [trace0,trace1]
    figure = go.Figure(data=data,layout=layout)

    return figure

# Breaking the frontend down into reusable modules for cleaner code.
# Omit external styling
def _omit(omit_keys, d):
    return {k: v for k,v in d.items() if k not in omit_keys}
# Customizable frontend for HTML card display
def CardForm(children,**kwargs):
    return html.Section(
        children,
        style=dict({
            'padding': 20,
            'margin': 5,
            'borderRadius': 5,
            'border': 'thin lightgrey solid',
            'font-family':'verdana'
        }, **(kwargs.get('style',{}))),
        **_omit(['style'], kwargs)
    )
# Inside card
# Customizable frontend for range sliders
def SliderForm(name,**kwargs):
    return html.Div(
        style={
            'padding':'20px 10px 25px 4px',
            'color':'white'
        },
        children=[
            html.P(f'{name}:',style={'font-weight':'bold'}),
            html.Div(
                style={
                    'margin-left': '6px'
                },
                children=dcc.Slider(**kwargs)
            )
        ]
    )
# Customizable frontend for dropdown menus
def DropdownForm(name,**kwargs):
    return html.Div(
        style={
            'margin': '10px 0px'
        },
        children=[
            html.P(
                children=f'{name}:',
                style={
                    'margin-left': '3px',
                    'font-weight':'bold',
                    'color':'white'
                },
            ),
            dcc.Dropdown(**kwargs)
        ]
    )
# Customizable frontend for radio buttons
def RadioButtonsForm(name,**kwargs):
    return html.Div(
        style={
            'padding': '20px 10px 25px 4px'
        },
        children=[
            html.P(f'{name}:',style={'font-weight':'bold'}),
            dcc.RadioItems(**kwargs)
        ]
    )


# Initializing instance
app = dash.Dash(__name__)
app.title = 'SVM Visual Tool'
server = app.server

# Laying out the frontend
app.layout = html.Div(children=[
    html.Div(className="banner",children=[
        # Set name
        html.Div(className="container scalable",children=[
            html.Br(),
            html.H1('Support Vector Machine(SVM) Visual Tool',style={'color':'#caf0f8','text-align':'center','font-family':'verdana','font-size':'36px'}),
            html.Br()
        ]),
    ]),
    # Body
    html.Div(id='body',className='container scalable',children=[
        # Viz area
        html.Div(className='row',children=[
            html.Div(
                className='three columns',
                id='div-viz',
                children=dcc.Graph(id='graph-svm')
            ), # div-graph will show the interactive/dynamic graphs via callacks,   
        ]),
        html.Div(className='row',children=[
            # Control panel : User board
            html.Div(
                className='three columns',
                style={
                    'width': '40%',
                    'height': '60%',
                    'display':'inline-block'
                },
                children=[
                    # Card element I
                    CardForm(children=[
                        html.H2("Dataset controls",style={'color':'white','font-weight':'bold'}),
                        # Select dataset type
                        DropdownForm(
                            name='Select Dataset Type',
                            id='dropdown-select-dataset-type',
                            options=[
                                {'label': 'Non Linear Type I', 'value': 'non_lin1'},
                                {'label': 'Non Linear Type II', 'value': 'non_lin2'},
                                {'label': 'Linear', 'value': 'lin'}
                            ],
                            clearable=False,
                            searchable=False,
                            value='non_lin2'    # Default
                        ),
                        SliderForm(
                            name='Size of Sample',
                            id='sample_size',
                            min=100,
                            max=800,
                            step=100,
                            # Marks in the slider
                            marks={i: str(i) for i in [100,200,300,400,500,600,700,800]},
                            value=300           # Default
                        ),
                        SliderForm(
                            name='Noise Level in data',
                            id='noise_slider',
                            min=0,
                            max=1,
                            marks={i/10: str(i/10) for i in range(0,12,2)},
                            step=0.1,
                            value=0.3,
                        ),
                    ],style={'backgroundColor':'#073b4c'}),
                    # Card element II
                    CardForm(children=[
                        html.H2("Threshold controls",style={'color':'white','font-weight':'bold'}),
                        # Select threshold slider
                        SliderForm(
                            name='Threshold',
                            id='select_threshold',
                            min=0,
                            max=1,
                            value=0.5,
                            step=0.01
                        ),
                        # Button to reset threshold
                        html.Button('Reset Threshold',id='thres_button'),
                    ],style={'backgroundColor':'#073b4c'}),
                    # Card element III (model hyperparameters)
                    CardForm(children=[
                        html.H2("Hyperparameter controls",style={'color':'white','font-weight':'bold'}),
                        # Kernel type
                        DropdownForm(
                            name='Choose Kernel Type',
                            id='svm-kernel-dropdown',
                            options=[
                                {'label': 'Linear', 'value': 'linear'},
                                {'label': 'Radial Basis Function(RBF)', 'value': 'rbf'},
                                {'label': 'Polynomial', 'value': 'poly'},
                                {'label': 'Sigmoid', 'value': 'sigmoid'}
                            ],
                            value='rbf',
                            clearable=False,
                            searchable=False
                        ),
                        # Hyperparameter C
                        SliderForm(
                            name='Choose C',
                            id='svm-c-slider',
                            min=-2,
                            max=4,
                            value=0,       # Set default to 10^0 = 1
                            marks={i: '{}'.format(10**i) for i in range(-2,5)}
                        ),
                        # Degree hyperparameter for polynomial kernel
                        SliderForm(
                            name='Choose Degree value',
                            id='svm-degree-slider',
                            min=2,
                            max=5,
                            value=2,       # Default degree = 2
                            step=1,
                            marks={i: str(i) for i in range(2, 6, 2)}
                        ),
                    ],style={'backgroundColor':'#073b4c'}),
                ]
            ),
            html.Div(
                className='three columns',
                style={
                    'width': '60%',
                    'height': '60%',
                    'display':'inline-block'
                },
                id='div-c',
                children=[
                    html.Br(),
                    CardForm(
                        children=[
                            html.H2('History of Support Vector Machines(SVM)',style={'color':'#001845'}),
                            html.P('In machine learning, support-vector machines (SVMs, also support-vector networks) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Developed at AT&T Bell Laboratories by Vapnik with colleagues (Boser et al., 1992, Guyon et al., 1993, Vapnik et al., 1997), it presents one of the most robust prediction methods, based on the statistical learning framework or VC theory proposed by Vapnik and Chervonekis (1974) and Vapnik (1982, 1995). Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier (although methods such as Platt scaling exist to use SVM in a probabilistic classification setting). The support-vector clustering algorithm, created by Hava Siegelmann and Vladimir Vapnik, applies the statistics of support vectors, developed in the support vector machines algorithm, to categorize unlabeled data, and is one of the most widely used clustering algorithms in industrial applications.',style={'font-size':'15px','font-weight':'bold','color':'#023e7d'}),
                            html.A(html.Button('WATCH THIS VIDEO TO GET AN INTUITION ABOUT SVMs',style={'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)','font-weight':'bold','color':'#001845','backgroundColor':'#84d2f6','font-size':'18px','height':'50px','width':'360px','border-radius':5,'border': 'thin deepblue solid'}),href='https://www.youtube.com/watch?v=efR1C6CvhmE',style={'padding-left':'26%'})
                        ],
                        style={'backgroundColor':'#93e1d8'}
                    ),
                    CardForm(
                        children=[
                            html.H2('Importance of hyperparameters C and kernel',style={'color':'#001845'}),
                            html.P('C : It is the regularization parameter that tells us, how much we want out SVM model to avoid misclassifications. So, as C increases, marginal threshold gets narower, as all misclassifications are handled perfectly. This gives us a Low Bias model (good on training data). Very high values will overfit the data. As C decreases, marginal threshold gets wider, as ample misclassifications are allowed. Very low values of C give us a High Bias model (poor on training data). But if a certain amount of misclassifications are allowed, the initial High Bias model gives a Low Variance model later (good performance on unseen data).',style={'font-size':'15px','font-weight':'bold','color':'#023e7d'}),
                            html.P('KERNEL : Say, we have non-linearly separable data that is in the form of two groups of concentric clusters. If we find a way to map the data from 2-dimensional space to 3-dimensional space, we will be able to find a decision surface that clearly divides the different classes. The first thought behind this data transformation process would be to map all the data points to a higher dimension (in this case, the 3rd dimension), and find the boundary hyperplane that results in an almost accurate classification. However, when there are more and more dimensions, computations within that space become more and more expensive. This is when the "kernel trick" comes in. It allows us to operate in the original feature space without computing the coordinates of the data in a higher dimensional space.',style={'font-size':'15px','font-weight':'bold','color':'#023e7d'}),
                            html.A(html.Button('READ : WHAT IS THE KERNEL TRICK ?',style={'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)','font-weight':'bold','color':'#001845','backgroundColor':'#84d2f6','font-size':'18px','height':'50px','width':'360px','border-radius':5,'border': 'thin deepblue solid'}),href='https://towardsdatascience.com/the-kernel-trick-c98cdbcaeb3f',style={'padding-left':'26%'})
                        ],
                        style={'backgroundColor':'#93e1d8'}
                    ),
                    CardForm(
                        children=[
                            html.H2('Developed by :',style={'color':'#93e1d8','text-align':'center'}),
                            html.Br(),
                            html.Div(
                                className='row',
                                children=[
                                    html.Div(
                                        className='three columns',
                                        children=[
                                            html.A(html.Button('BALAKA BISWAS',style={'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19)','height':'60px','font-weight':'bold','color':'#001845','backgroundColor':'#84d2f6','font-size':'18px','border-radius':5,'border': 'thin deepblue solid'}),href='https://www.linkedin.com/in/balaka-biswas/'),
                                        ],
                                        style={
                                            'display':'inline-block',
                                            'padding-left':'38.5%'
                                        }
                                    )
                                ]
                            )
                        ]
                    )
                ]
            )   
        ])
    ])
])

# Backend
# Reset threshold button
@app.callback(Output('select_threshold','value'),
            [Input('thres_button','n_clicks')],
            [State('graph-svm','figure')])
def reset_threshold(n_clicks,figure):
    if n_clicks:
        z_ax = np.array(figure['data'][0]['z'])   # Extracting the Z-axis data
        value = - z_ax.min()/(z_ax.max() - z_ax.min())
    else:
        value = 0.50
    return value
# Disable the Degree slider if polynomial kernel is not selected
@app.callback(Output('svm-degree-slider','disabled'),
            [Input('svm-kernel-dropdown','value')])
def disable_degree(kernel):
    return kernel != 'poly'
# Main plot
@app.callback(Output('div-viz','children'),
            [Input('dropdown-select-dataset-type','value'),
            Input('sample_size','value'),
            Input('noise_slider','value'),
            Input('select_threshold','value'),
            Input('svm-kernel-dropdown','value'),
            Input('svm-c-slider','value'),
            Input('svm-degree-slider','value')])
def plot_update(dataset, sample_size, noise, threshold, kernel, C_pow, degree):
    # Data
    X,Y = gen_dat(n_samples=sample_size,noise=noise,dataset_type=dataset)
    # Scaling
    sc = StandardScaler()
    X = sc.fit_transform(X)
    trainx,testx,trainy,testy = train_test_split(X,Y,test_size=0.3,random_state=36)
    # Set limits for mesh grid
    minx = X[:,0].min() - 0.5
    maxx = X[:,0].max() + 0.5
    miny = X[:,1].min() - 0.5
    maxy = X[:,1].max() + 0.5
    # Create mesh grid
    XX,YY = np.meshgrid(np.arange(minx,maxx,0.3),np.arange(miny,maxy,0.3))
    # Setting the value of C
    C = 10**C_pow
    # Training the model
    svc = SVC(
        C = C,
        kernel= kernel,
        degree= degree
    )
    svc.fit(trainx,trainy)
    # Decision boundary
    if hasattr(svc, "decision_function"):
        Z = svc.decision_function(np.c_[XX.ravel(),YY.ravel()])
    else:
        Z = svc.predict_proba(np.c_[XX.ravel(),YY.ravel()])[:,1]

    # Plot 1 : SVM Main plot
    main_plot = plot_svm(
        model=svc,
        trainx=trainx,
        testx=testx,
        trainy=trainy,
        testy=testy,
        Z = Z,
        XX=XX,
        YY=YY,
        mesh_step = 0.3,
        threshold = threshold

    )
    # Plot 2 : 
    # (a) Confusion matrix 
    conf_pie = pie_conf(
        model=svc,
        testx=testx,
        testy=testy,
        Z=Z,
        threshold=threshold
    )
    # (b) 3D plot
    plot3D = plot_gaus3d(
        model=svc,
        X=X,
        Y=Y,
        type='gaussian'
    )

    # Display plot
    if kernel != 'rbf':
        return [
            html.Div(
                className='two columns',
                style={'width':'100%','height':'calc(100vh-90px)','margin-top':'5px'},
                children=[
                    dcc.Graph(
                        id='graph-pieconf',
                        style={'height':'90vh','width':'20%','display':'inline-block'},
                        figure=conf_pie
                    ),
                    dcc.Graph(
                        id='graph-svm',
                        figure=main_plot,
                        style={'height':'90vh','width':'79%','display':'inline-block'}
                    )
                ]
            ),
        ]
    else:
        return [
            html.Div(
                className='two columns',
                style={'width':'100%','height':'calc(100vh-90px)','margin-top':'5px'},
                children=[
                    dcc.Graph(
                        id='graph-plot3d',
                        style={'height':'90vh','width':'50%','display':'inline-block'},
                        figure=plot3D
                    ),
                    dcc.Graph(
                        id='graph-svm',
                        figure=main_plot,
                        style={'height':'90vh','width':'50%','display':'inline-block'}
                    )
                ]
            ),
        ]


# Run
if __name__ == "__main__":
    app.run_server(debug=True)