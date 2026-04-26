
import flet as ft
import sys
import os

# Import des modules du dossier backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from database_manager import Manager
from graphs_visualizer import Graph_Visualization

def main(page: ft.Page):
    page.title = "Vora - Dashboard Analytique"
    page.theme_mode = "dark" # Correction pour compatibilité ancienne version
    page.padding = 20
    page.scroll = "auto"
    
    db = Manager(dbname="test_data.db")
    viz = Graph_Visualization(db)

    def create_chart_container(title, img_base64):
        return ft.Container(
            content=ft.Column([
                ft.Text(title, size=18, weight="bold", color="#0078D4"),
                # Dans Flet 0.84, on utilise 'src' pour le base64 et la chaîne 'contain'
                ft.Image(src=img_base64, fit="contain") 
            ]),
            bgcolor="#1C1C1C",
            padding=15,
            border_radius=10,
            expand=True,
            border=ft.border.all(1, "#333333")
        )

    # Récupération des images via le visualizer
    img_hist = viz.get_histogram_img(db.get_histogram_query("alltime"), "Distribution des clignements (Intervalles de 3)")
    img_low = viz.get_low_freq_weekly_img(db.get_low_freq_weekly_query(7))
    img_scatter = viz.get_scatter_session_img(db.get_scatter_session_query())

    # Construction de la page
    page.add(
        ft.Row([
            ft.Text("STATISTIQUES VORA", size=32, weight="bold"),
        ], alignment="center"),
        
        ft.Divider(height=20, color="transparent"),
        
        # Première ligne : Histogramme et Fatigue
        ft.Row([
            create_chart_container("Historique Global (Reliable)", img_hist),
            create_chart_container("Minutes < 10 (7 jours)", img_low),
        ], height=400),
        
        ft.Divider(height=20, color="transparent"),
        
        # Deuxième ligne : Session actuelle
        ft.Row([
            create_chart_container("Nuage de points : Session Actuelle", img_scatter)
        ], height=450)
    )

if __name__ == "__main__":
    # ft.app reste la méthode la plus robuste sur Arch pour cette version
    ft.app(target=main)
