from fonctions import build_graph_d_w


if __name__ == "__main__":
    # Modifier les paramètres pour changer le graphique produit.
    build_graph_d_w(
                        list_d=[0, 2, 3, 4, 8, 30],
                        C=1,
                        nb_simulation=200,
                        nb_data=180,
                        dark_mode=True,
                        seed=1,
                    )