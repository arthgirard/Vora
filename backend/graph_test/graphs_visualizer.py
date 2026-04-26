
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

class Graph_Visualization:
    def __init__(self, manager):
        self.manager = manager
        # Style sombre pour l'intégration logicielle
        plt.style.use('dark_background')
        plt.rcParams.update({
            "figure.facecolor": "#1C1C1C",
            "axes.facecolor": "#1C1C1C",
            "grid.color": "#444444"
        })

    def fig_to_base64(self, fig):
        """Convertit une figure Matplotlib en chaîne base64 compatible Flet 0.84."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def get_histogram_img(self, query, title="Distribution"):
        """Génère l'histogramme des intervalles de 3 clignements."""
        df = pd.read_sql_query(query, self.manager.connect)
        fig, ax = plt.subplots(figsize=(5, 4))
        if not df.empty:
            max_val = df['blink_count'].max()
            bins = np.arange(0, max_val + 4, 3)
            ax.hist(df['blink_count'], bins=bins, color='#0078D4', edgecolor='white')
            ax.set_title(title)
            ax.set_xlabel("Clignements / min")
            ax.set_ylabel("Fréquence")
        return self.fig_to_base64(fig)

    def get_low_freq_weekly_img(self, query):
        """Génère l'histogramme de fatigue (low_freq) sur 7 jours."""
        df = pd.read_sql_query(query, self.manager.connect)
        fig, ax = plt.subplots(figsize=(5, 4))
        if not df.empty:
            ax.bar(df['x_plot'], df['y_plot'], color='#FF5252')
            ax.set_title("Alertes Fatigue (7 derniers jours)")
            plt.xticks(rotation=45)
        return self.fig_to_base64(fig)

    def get_scatter_session_img(self, query):
        """Génère le nuage de points pour la session en cours."""
        df = pd.read_sql_query(query, self.manager.connect)
        fig, ax = plt.subplots(figsize=(10, 4))
        if not df.empty:
            ax.scatter(df['x_plot'], df['y_plot'], color='#4CAF50', alpha=0.6)
            ax.set_title("Progression de la Session")
            ax.set_xlabel("Minute")
            ax.set_ylabel("Clignements")
        return self.fig_to_base64(fig)
