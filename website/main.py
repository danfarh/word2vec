import pickle
import numpy as np
import plotly.graph_objs as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances


# load embedding weights
with open('data/embedding_weights.h5', 'rb') as f:
    embedding_weights = pickle.load(f)
# load tokenizer
with open('data/tokenizer.h5', 'rb') as f:
    tokenizer = pickle.load(f)

# Dimention Reduction -> PCA
three_dim = PCA(random_state=0).fit_transform(embedding_weights)[:, :3]
print(three_dim.shape)

# unique words
words = [word for word in tokenizer.word_index.keys()]

# Get most similar words
def get_most_similarity(word, weights=three_dim, tokenizer=tokenizer, n=20):
    distance_matrix = euclidean_distances(weights)
    word_to_sequences = tokenizer.texts_to_sequences([word])[0][0]
    index = np.argsort(distance_matrix[word_to_sequences])[:n]
    return index

st.sidebar.subheader('Word Embeddings')
one_or_all_words = st.sidebar.selectbox(
    'Watch all words or one word map? ', ('ONE_WORD', 'ALL_WORDS'))

if one_or_all_words == 'ONE_WORD':
    word_input = st.sidebar.text_input("Enter a word: ", '')

    if not word_input:
        word_input = 'مه'

    word_index = get_most_similarity(word_input)
    indexes = [index for index in word_index]
    #print(indexes)

# Frontend
color = 'blue'
quiver = go.Cone(
    x=[0, 0, 0],
    y=[0, 0, 0],
    z=[0, 0, 0],
    u=[1.5, 0, 0],
    v=[0, 1.5, 0],
    w=[0, 0, 1.5],
    anchor="tail",
    colorscale=[[0, color], [1, color]],
    showscale=False
)
data = [quiver]

# One word plot
if one_or_all_words == 'ONE_WORD':
    for i in range(len(indexes)):
        trace = go.Scatter3d(
            x=three_dim[indexes[i]:indexes[i]+1, 0],
            y=three_dim[indexes[i]:indexes[i]+1, 1],
            z=three_dim[indexes[i]:indexes[i]+1, 2],
            text=words[indexes[i]:indexes[i]+1],
            name=word_input,
            textposition="top center",
            textfont_size=30,
            mode='markers+text',
            marker={
                'size': 10,
                'opacity': 0.8,
                'color': 2
            }
        )
        data.append(trace)
        # Configure the layout
        layout = go.Layout(
            margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
            showlegend=True,
            legend=dict(
                x=1,
                y=0.5,
                font=dict(
                    family="Courier New",
                    size=25,
                    color="black"
                )),
            font=dict(
                family=" Courier New ",
                size=15),
            autosize=False,
            width=1000,
            height=1000
        )
    # draw plot
    plot_figure = go.Figure(data=data, layout=layout)
    st.plotly_chart(plot_figure)


# All words plot
if one_or_all_words == 'ALL_WORDS':
    count = 0
    trace_input = go.Scatter3d(
        x=three_dim[count:1000, 0],
        y=three_dim[count:1000, 1],
        z=three_dim[count:1000, 2],
        text=words[count:],
        name='all words',
        textposition="top center",
        textfont_size=30,
        mode='markers+text',
        marker={
            'size': 10,
            'opacity': 1,
            'color': 'green'
        }
    )
    data.append(trace_input)
    # Configure the layout
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
            x=1,
            y=0.5,
            font=dict(
                family="Courier New",
                size=25,
                color="green"
            )),
        font=dict(
            family=" Courier New ",
            size=15),
        autosize=False,
        width=1000,
        height=1000
    )
    # draw plot
    plot_figure = go.Figure(data=data, layout=layout)
    st.plotly_chart(plot_figure)
