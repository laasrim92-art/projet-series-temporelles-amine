import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.title("Analyse et Prédiction de Séries Temporelles")
st.markdown("""
Cette application permet de visualiser, analyser et prédire une série temporelle.
**Auteur :** Radja KURNIAWAN
""")

st.sidebar.header("1. Chargement des données")
uploaded_file = st.sidebar.file_uploader("Charge le fichier CSV ou Excel", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        # Lecture du fichier selon le format
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success("Fichier chargé avec succès !")
        
        st.sidebar.subheader("Configuration des colonnes")
        all_columns = df.columns.tolist()
        
        col_date = st.sidebar.selectbox("Choisissez la colonne Date", all_columns)
        col_value = st.sidebar.selectbox("Choisissez la colonne Valeur (Série à prédire)", all_columns)
        
        try:
            df[col_date] = pd.to_datetime(df[col_date])
            df = df.sort_values(by=col_date)
            df.set_index(col_date, inplace=True)
            ts = df[col_value] # La série temporelle
            
            st.header("2. Visualisation des données")
            st.subheader("Série originale")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(ts)
            ax.set_title("Évolution temporelle")
            st.pyplot(fig)
            
            st.subheader("Décomposition STL (Tendance, Saisonnalité, Résidus)")
            # Le paramètre period est important, on laisse l'utilisateur ajuster si besoin
            period = st.sidebar.slider("Périodicité (ex: 12 pour mois, 7 pour jours)", 1, 365, 12)
            res = seasonal_decompose(ts, period=period, model='additive', extrapolate_trend='freq')
            fig_stl = res.plot()
            st.pyplot(fig_stl)

            st.subheader("Test de Stationnarité (ADF)")
            adf_result = adfuller(ts.dropna())
            st.write(f"**ADF Statistic:** {adf_result[0]:.4f}")
            st.write(f"**p-value:** {adf_result[1]:.4f}")
            if adf_result[1] < 0.05:
                st.success("La série est stationnaire (p-value < 0.05).")
            else:
                st.warning("La série n'est pas stationnaire (p-value >= 0.05).")

            st.header("3. Modélisation et Prédiction")
            model_type = st.radio("Choisissez le modèle", ["ARIMA", "SARIMA"])
            
            col1, col2, col3 = st.columns(3)
            p = col1.number_input("p (AR)", 0, 10, 1)
            d = col2.number_input("d (I)", 0, 5, 1)
            q = col3.number_input("q (MA)", 0, 10, 1)
            
            if model_type == "SARIMA":
                st.markdown("Paramètres Saisonniers")
                scol1, scol2, scol3, scol4 = st.columns(4)
                P = scol1.number_input("P", 0, 5, 1)
                D = scol2.number_input("D", 0, 5, 1)
                Q = scol3.number_input("Q", 0, 5, 1)
                s = scol4.number_input("s (Saisonnalité)", 0, 50, 12)
                order = (p, d, q)
                seasonal_order = (P, D, Q, s)
            else:
                order = (p, d, q)
                seasonal_order = None

            if st.button("Entraîner le modèle"):
                with st.spinner("Entraînement en cours..."):
                    if model_type == "ARIMA":
                        model = ARIMA(ts, order=order)
                    else:
                        model = SARIMAX(ts, order=order, seasonal_order=seasonal_order)
                    
                    model_fit = model.fit()
                    st.success("Modèle entraîné !")
                    
                    horizon = st.number_input("Horizon de prédiction (nombre de pas)", 1, 100, 12)
                    
                    forecast = model_fit.get_forecast(steps=horizon)
                    forecast_index = pd.date_range(start=ts.index[-1], periods=horizon+1, freq=pd.infer_freq(ts.index))[1:]
                    
                    st.header("4. Résultats de la prédiction")
                    
                    pred_series = pd.Series(forecast.predicted_mean.values, index=forecast_index)
                    
                    fig_pred, ax_pred = plt.subplots(figsize=(10, 5))
                    ax_pred.plot(ts, label="Observé")
                    ax_pred.plot(pred_series, label="Prédiction", color='red')
                    ax_pred.legend()
                    st.pyplot(fig_pred)
                    
                    st.write("Valeurs prédites :")
                    st.dataframe(pred_series)

        except Exception as e:
            st.error(f"Erreur lors du traitement des données : {e}")
            st.info("Vérifiez que la colonne Date est bien au format date et que la colonne Valeur est numérique.")

    except Exception as e:
        st.error(f"Erreur de lecture du fichier : {e}")
