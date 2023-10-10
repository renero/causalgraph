import math
import numpy as np
import pandas as pd
import networkx as nx


class Knowledge:
    """
    This class collects everything we know about each edge in the proposed graph
    in terms of the following properties:

    - origin: the origin node
    - target: the target node
    - ref_edge: whether the edge is in the reference graph
    - correlation: the correlation between the individual SHAP values and the origin node
    - KS_pval: the p-value of the Kolmogorov-Smirnov test between the origin and the target
    - shap_edge: whether the edge is in the graph constructed after evaluating mean 
        SHAP values.
    - shap_skedastic_pval: the p-value of the skedastic test for the SHAP values
    - parent_skedastic_pval: the p-value of the skedastic test for the parent values
    - mean_shap: the mean of the SHAP values between the origin and the target
    - slope_shap: the slope of the linear regression for target vs. SHAP values
    - slope_target: the slope of the linear regression for the target vs. origin values

    """

    def __init__(self, rex: object, ref_graph: nx.DiGraph):
        """
        Arguments:
        ----------
            shaps (ShapEstimator): The shap estimator.
            ref_graph (nx.DiGraph): The reference graph, or ground truth.    
        """
        assert rex is not None, "Rex is None"
        assert rex.hierarchies is not None, "Hierarchies is None"
        assert rex.shaps is not None, "ShapEstimator is None"
        assert rex.pi is not None, "PIEstimator is None"

        self.K = 180.0 / math.pi
        self.shaps = rex.shaps
        self.pi = rex.pi
        self.hierarchies = rex.hierarchies
        self.feature_names = rex.feature_names
        self.scoring = rex.models.scoring
        self.ref_graph = ref_graph
        self.G_shap = rex.G_shap
        self.root_causes = rex.root_causes
        
        self.correlation_th = rex.correlation_th
        if self.correlation_th is not None:
            self.correlated_features = self.hierarchies.correlated_features

    def data(self):
        """Returns a dataframe with the knowledge about each edge in the graph"""
        rows = []
        for origin in self.feature_names:
            for target in self.feature_names:
                if origin == target:
                    continue

                if self.correlation_th is not None:
                    if target in self.correlated_features[origin]:
                        continue

                if self.correlation_th is not None:
                    all_features = [f for f in self.feature_names if (
                        f != origin) and (f not in self.correlated_features[origin])]
                else:
                    all_features = [f for f in self.feature_names if f != origin]
                feature_pos = all_features.index(target)

                sd = self.shaps.shap_discrepancies[origin][target]
                pi = self.pi.pi[origin]['importances_mean'][feature_pos]

                b0_s, beta1_s = sd.shap_model.params[0], sd.shap_model.params[1]
                b0_y, beta1_y = sd.parent_model.params[0], sd.parent_model.params[1]
                shap_slope = math.atan(beta1_s)*self.K
                parent_slope = math.atan(beta1_y)*self.K
                rows.append({
                    'origin': origin,
                    'target': target,
                    'ref_edge': int((origin, target) in self.ref_graph.edges()),
                    'correlation': self.hierarchies.correlations[origin][target],
                    'shap_correlation': sd.shap_correlation,
                    'KS_pval': sd.ks_pvalue,
                    'shap_edge': int(origin in set(self.G_shap.predecessors(target))),
                    'shap_skedastic_pval': sd.shap_p_value,
                    'parent_skedastic_pval': sd.parent_p_value,
                    'mean_shap': self.shaps.shap_mean_values[origin][feature_pos],
                    'mean_pi': pi,
                    'slope_shap': shap_slope,
                    'slope_target': parent_slope,
                    'potential_root': int(origin in self.root_causes)
                })
        self.results = pd.DataFrame.from_dict(rows)
        return self.results


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from causalgraph.common.utils import graph_from_dot_file, load_experiment
    from sklearn.preprocessing import StandardScaler

    # Display Options
    np.set_printoptions(precision=4, linewidth=100)
    pd.set_option('display.precision', 4)
    pd.set_option('display.float_format', '{:.4f}'.format)

    # Paths
    path = "/Users/renero/phd/data/RC3/"
    output_path = "/Users/renero/phd/output/RC3/"
    # experiment_name = 'rex_generated_linear_1'
    experiment_name = 'custom_rex'

    # Read the data
    ref_graph = graph_from_dot_file(f"{path}{experiment_name}.dot")
    data = pd.read_csv(f"{path}{experiment_name}.csv")
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

    # Split the dataframe into train and test
    train = data.sample(frac=0.9, random_state=42)
    test = data.drop(train.index)

    custom = load_experiment(f"{experiment_name}", output_path)
    custom.is_fitted_ = True
    print(f"Loaded experiment {experiment_name}")

    custom.feature_names = list(data.columns)
    custom.models.score(train)
    custom.knowledge(ref_graph)
