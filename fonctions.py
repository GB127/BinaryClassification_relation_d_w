import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from sklearn import svm
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import DecisionBoundaryDisplay
import warnings


def generate_two_distributions(nb_data:int, distance:float)->tuple[np.ndarray]:
    """Génère deux ensembles de données suivant une loi normale. L'écart entre les moyennes des deux lois normales correspond à la distance spécifiée.

    Args:
        nb_data (int): Quantité de coordonnées (x1, x2) à générer par ensemble.
        alpha (float): Distance entre la moyenne des deux distributions.

    Returns:
        (np.ndarray, np.ndarray): les deux ensembles générés
    """
    cov = [[1, 0], [0, 1]]  # Pour une distribution circulaire.
    X1 = np.random.multivariate_normal([0, 0], cov, nb_data)
    X2= np.random.multivariate_normal([distance, 0], cov, nb_data)
    return X1, X2

def build_predictor_SVM(X1:np.ndarray, X2:np.ndarray, C:int=1)->svm.LinearSVC:
    """Utilise l'algorithme d'apprentissage SVM implémenté par sklearn pour produire un prédicteur (classification binaire).

    Args:
        X1 (np.ndarray): 1er ensemble de données.
        X2 (np.ndarray): 2e ensemble de données
        C (int): Paramètre de régularisation à utiliser pour l'algorithme SVM.

    Returns:
        svm.LinearSVC: Prédicteur trouvé
    """
    # Préparation des arguments à fournir pour l'algorithme svm.LinearSVC.
    X12 = np.vstack((X1, X2))
    y = np.zeros((X12.shape[0]))
    y[:X1.shape[0]] = 1

    # Application de l'algorithme.
    clf = svm.LinearSVC(C=C, loss="hinge")
    # Ignorer les avertissements ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    clf.fit(X12, y)
    return clf


def find_closest_distances(distance_depart:list[int | float], all_distances:np.ndarray)-> list[int|float]:
    """Trouve les distances se rapprochant le plus aux distances de départ.

    Args:
        distance_depart (list[int  |  float]): Distances à "matcher"
        all_distances (np.ndarray): Toutes les distances.

    Returns:
        list: les distances se rapprochant le plus aux distances de départ spécifiées.
    """
    closest_distances = []
    for d in distance_depart:
        index_value = np.abs((all_distances - d)).argmin()
        closest_distances.append(all_distances[index_value])
    return closest_distances


def build_graph_d_w(list_d:list[int | float], C:float=1, nb_simulation:int=200, nb_data:int=100,dark_mode=False, seed=False):
    """Construit un graphe représentant la relation entre la distance séparant deux distributions normales et la norme du classificateur binaire trouvé par l'algorithme SVM.

    Args:
        list_d (list[int  |  float]): distances à afficher à la droite du graphique. Le minimum et le maximum des distances déterminent la plage de valeur de distances (abcisses du graphique)
        C (float, optional): Paramètre de régularisation à utiliser pour L'algorithme SVM. Defaults to 1.
        nb_simulation (int, optional): Nombre de simulations à produire (nombre de points dans le graphique). Defaults to 200.
        nb_data (int, optional): Nombre de points par distribution normale. Defaults to 100.
        dark_mode (bool): Modifie le thème du graphique.
        seed (any): Seed pour la randomization.

    Raises:
        ValueError: Il faut au moins 2 distances dans list_d.
    """
    if seed:
        np.random.seed(seed)
    if len(list_d) == 1:
        raise ValueError("Il faut au moins 2 distances dans list_d.")
    elif len(list_d) > 8:
        warnings.warn(f"Trop de distributions ({len(list_d)}) à afficher. Il est conseillé d'afficher au plus 8 distributions.")

    if dark_mode:  # Selon mes expérimentations, on voit beaucoup mieux sur un graphique noir.
        cmap = cm.get_cmap('spring_r', len(list_d))
        plt.style.use('dark_background')
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[mcolors.to_hex(cmap(i)) for i in range(len(list_d))])

    # Préparer la disposition des graphiques.
    fig, axs = plt.subplots(ncols=2, nrows=len(list_d), gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle("Relation entre la distance (d) séparant deux distributions\n normales et la norme du classificateur binaire ($w$).")
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    # Préparation du graphique de gauche.
    for ax in axs[:, 0]:
        ax.remove()
    gs = axs[0, 0].get_gridspec()
    ax_graph = fig.add_subplot(gs[:, 0])

    # Éliminer les doublons et trier les distances en ordre croissant.
    list_d = sorted(set(list_d))

    # Préparer les valeurs de l'abcisse à utiliser.
    all_distances = np.linspace(min(list_d), max(list_d), nb_simulation)
    # Garder en mémoire les distances les plus proches des valeurs spécifiées.
    closest_ds = find_closest_distances(list_d, all_distances)

    for d in all_distances:
        distribution_1, distribution_2 = generate_two_distributions(nb_data=nb_data, distance=d)
        predictor_w = build_predictor_SVM(distribution_1, distribution_2, C=C)  # Objet représentant le prédicteur trouvé.
        w = predictor_w.coef_[0]  # w : prédicteur (vecteur : [w1, w2])
        norm_w = np.linalg.norm(w)

        if d not in closest_ds:
            ax_graph.scatter(d, norm_w, s=5, color="white" if dark_mode else "black", alpha=0.4)
            continue  # On n'a pas besoin d'afficher la distribution à la droite, on passe au suivant.
        # Affichage du point sur le graphique.
        ax_graph.scatter(d, norm_w, s=60, marker="*", label=f"d = {round(d, 2)}")

        # On affiche la distribution à droite.
        no_graph = closest_ds.index(d)  # Numéro du graph de distribution.
        axs[no_graph,1].set_xticks([])
        axs[no_graph,1].set_yticks([])
        axs[no_graph,1].scatter(distribution_1[:, 0], distribution_1[:, 1], marker=".", alpha=0.2, color="limegreen" if dark_mode else "blue")
        axs[no_graph,1].scatter(distribution_2[:, 0], distribution_2[:, 1], marker=".", alpha=0.2, color="orange" if dark_mode else "red")
        # Tracer du prédicteur : Solution inspirée et adaptée de https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html#sphx-glr-auto-examples-svm-plot-separating-hyperplane-py
        DecisionBoundaryDisplay.from_estimator(
                                            predictor_w,
                                            np.vstack((distribution_1, distribution_2)),
                                            plot_method="contour",
                                            colors="white" if dark_mode else "black",
                                            levels=[0],
                                            alpha=1,
                                            linestyles=["-"],
                                            linewidths=2,
                                            ax=axs[no_graph,1],
                                    )
        
        # Affichage de la distance sur la représentation de la distribution.
        axs[no_graph,1].text(0.95, 0.05,
                                f"d = {round(d,2)}",
                                transform=axs[no_graph,1].transAxes,
                                ha='right', va='bottom',
                                fontsize=10,
                                bbox=dict(facecolor='black' if dark_mode else "white", edgecolor='gray')
                            )


    ax_graph.legend()
    ax_graph.set_ylabel("$||w||$")
    ax_graph.set_xlabel("d")

    plt.savefig("relation_d_w.png")
    print("Graphique sauvegardé!")
