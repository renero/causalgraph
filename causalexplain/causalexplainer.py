"""
This module contains the GraphDiscovery class which is responsible for
creating, fitting, and evaluating causal discovery experiments.
"""
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

from causalexplain.common import (
    DEFAULT_REGRESSORS,
    utils,
)
from causalexplain.common import plot
from causalexplain.common.notebook import Experiment
from causalexplain.metrics.compare_graphs import evaluate_graph


class GraphDiscovery:
    def __init__(
        self,
        experiment_name: str = None,
        model_type: str = 'rex',
        csv_filename: str = None,
        true_dag_filename: str = None,
        verbose: bool = False,
        seed: int = 42
    ) -> None:
        """
        Initializes a new instance of the GraphDiscovery class.

        Args:
            experiment_name (str, optional): The name of the experiment.
            model_type (str, optional): The type of model to use. Valid options
                are: 'rex', 'pc', 'fci', 'ges', 'lingam', 'cam', 'notears'.
            csv_filename (str, optional): The filename of the CSV file containing
                the data.
            true_dag_filename (str, optional): The filename of the DOT file
                containing the true causal graph.
            verbose (bool, optional): Whether to print verbose output.
            seed (int, optional): The random seed for reproducibility.
        """
        # Normalize empty/whitespace strings to None
        experiment_name = experiment_name.strip() if isinstance(experiment_name, str) else experiment_name
        experiment_name = None if experiment_name == "" else experiment_name
        csv_filename = csv_filename.strip() if isinstance(csv_filename, str) else csv_filename
        csv_filename = None if csv_filename == "" else csv_filename

        if (experiment_name is None and csv_filename is not None) or \
                (experiment_name is not None and csv_filename is None):
            raise ValueError(
                f"Both 'experiment_name' and 'csv_filename' must be provided together, "
                f"or none of them. Got experiment_name='{experiment_name}', "
                f"csv_filename='{csv_filename}'")
        elif experiment_name is None and csv_filename is None:
            self.experiment_name = None
            self.estimator = 'rex'
            self.csv_filename = None
            self.dot_filename = None
            self.verbose = False
            self.seed = 42
            return

        self.experiment_name = experiment_name
        self.estimator = model_type
        self.csv_filename = csv_filename
        self.dot_filename = true_dag_filename
        self.verbose = verbose
        self.seed = seed

        self.dataset_path = os.path.dirname(csv_filename)
        self.output_path = os.getcwd()
        self.trainer = {}

        # Read the reference graph
        self.ref_graph = utils.graph_from_dot_file(true_dag_filename)

        # assert that the data file exists
        if not os.path.exists(csv_filename):
            raise FileNotFoundError(f"Data file {csv_filename} not found")
        self.dataset_name = os.path.splitext(os.path.basename(csv_filename))[0]

        # Read the column names of the data.
        data = pd.read_csv(csv_filename)
        self.data_columns = list(data.columns)
        del data

        if self.estimator == 'rex':
            self.regressors = DEFAULT_REGRESSORS
        else:
            self.regressors = [self.estimator]

    def create_experiments(self) -> dict:
        """
        Create an Experiment object for each regressor.

        Args:
            dataset_name (str): Name of the dataset
            true_dag (str): Path to the true DAG DOT file
            regressors (list): List of regressor types to create experiments for
            dataset_path (str): Path to the input dataset
            output_path (str): Path for output files

        Returns:
            dict: A dictionary of Experiment objects
        """
        self.trainer = {}
        for model_type in self.regressors:
            trainer_name = f"{self.dataset_name}_{model_type}"
            self.trainer[trainer_name] = Experiment(
                experiment_name=self.dataset_name,
                csv_filename=self.csv_filename,
                dot_filename=self.dot_filename,
                model_type=model_type,
                input_path=self.dataset_path,
                output_path=self.output_path,
                verbose=False)

        return self.trainer

    def fit_experiments(
        self,
        hpo_iterations: int = None,
        bootstrap_iterations: int = None,
        **kwargs
    ) -> None:
        """
        Fit the Experiment objects.

        Args:
            trainer (dict): A dictionary of Experiment objects
            estimator (str): The estimator to use ('rex' or other)
            verbose (bool, optional): Whether to print verbose output. 
                Defaults to False.
            hpo_iterations (int, optional): Number of HPO trials for REX. 
                Defaults to None.
            bootstrap_iterations (int, optional): Number of bootstrap trials 
                for REX. Defaults to None.
        """
        if self.estimator == 'rex':
            xargs = {
                'verbose': self.verbose,
                'hpo_n_trials': hpo_iterations,
                'bootstrap_trials': bootstrap_iterations
            }
        else:
            xargs = {
                'verbose': self.verbose
            }

        # Combine the arguments
        xargs.update(kwargs)

        for trainer_name, experiment in self.trainer.items():
            if not trainer_name.endswith("_rex"):
                experiment.fit_predict(estimator=self.estimator, **xargs)

    def combine_and_evaluate_dags(self) -> Experiment:
        """
        Retrieve the DAG from the Experiment objects.

        Returns:
            Experiment: The experiment object with the final DAG
        """
        if self.estimator != 'rex':
            trainer_key = f"{self.dataset_name}_{self.estimator}"
            estimator_obj = getattr(self.trainer[trainer_key], self.estimator)
            self.trainer[trainer_key].dag = estimator_obj.dag
            if self.ref_graph is not None and self.data_columns is not None:
                self.trainer[trainer_key].metrics = evaluate_graph(
                    self.ref_graph, estimator_obj.dag, self.data_columns)
            else:
                self.trainer[trainer_key].metrics = None

            self.dag = self.trainer[trainer_key].dag
            self.metrics = self.trainer[trainer_key].metrics
            return self.trainer[trainer_key]

        # For ReX, we need to combine the DAGs. Hardcode for now to combine
        # the first and second DAGs
        estimator1 = getattr(self.trainer[list(self.trainer.keys())[0]], 'rex')
        estimator2 = getattr(self.trainer[list(self.trainer.keys())[1]], 'rex')
        _, _, dag, _ = utils.combine_dags(estimator1.dag, estimator2.dag,
                                          estimator1.shaps.shap_discrepancies)

        # Create a new Experiment object for the combined DAG
        new_trainer = f"{self.dataset_name}_rex"
        if new_trainer in self.trainer:
            del self.trainer[new_trainer]
        self.trainer[new_trainer] = Experiment(
            experiment_name=self.dataset_name,
            model_type='rex',
            input_path=self.dataset_path,
            output_path=self.output_path,
            verbose=False)

        # Set the DAG and evaluate
        self.trainer[new_trainer].ref_graph = self.ref_graph
        self.trainer[new_trainer].dag = dag
        if self.ref_graph is not None and self.data_columns is not None:
            self.trainer[new_trainer].metrics = evaluate_graph(
                self.ref_graph, dag, self.data_columns)
        else:
            self.trainer[new_trainer].metrics = None

        self.dag = self.trainer[new_trainer].dag
        self.metrics = self.trainer[new_trainer].metrics
        return self.trainer[new_trainer]

    def run(
            self,
            hpo_iterations: int = None,
            bootstrap_iterations: int = None,
            **kwargs):
        """
        Run the experiment.

        Args:
            hpo_iterations (int, optional): Number of HPO trials for REX. 
                Defaults to None.
            bootstrap_iterations (int, optional): Number of bootstrap trials 
                for REX. Defaults to None.
        """
        self.create_experiments()
        self.fit_experiments(hpo_iterations, bootstrap_iterations, **kwargs)
        self.combine_and_evaluate_dags()

    def save(self, full_filename_path: str) -> None:
        """
        Save the model as an Experiment object.

        Args:
            full_filename_path (str): A full path where to save the model,
                including the filename.
        """
        assert self.trainer, "No trainer to save"
        assert full_filename_path, "No output path specified"

        trainer_name = os.path.splitext(
            os.path.basename(full_filename_path))[0]
        full_dir_path = os.path.dirname(full_filename_path)
        assert os.path.exists(full_dir_path), \
            f"Output directory {full_dir_path} does not exist"

        saved_as = utils.save_experiment(
            trainer_name, full_dir_path, self.trainer, overwrite=False)
        print(f"Saved model as: {saved_as}", flush=True)

    def load(self, model_path: str) -> Experiment:
        """
        Load the model from a pickle file.

        Args:
            model_path (str): Path to the pickle file containing the model

        Returns:
            Experiment: The loaded Experiment object
        """
        with open(model_path, 'rb') as f:
            self.trainer = pickle.load(f)
            print(f"Loaded model from: {model_path}", flush=True)

        # Set the dag and metrics
        self.dag = self.trainer[list(self.trainer.keys())[-1]].dag
        self.metrics = self.trainer[list(self.trainer.keys())[-1]].metrics
        return self.trainer

    def printout_results(self, graph, metrics):
        """
        This method prints the DAG to stdout in hierarchical order.

        Parameters:
        -----------
        dag : nx.DiGraph
            The DAG to be printed.
        """
        if len(graph.edges()) == 0:
            print("Empty graph")
            return

        print("Resulting Graph:\n---------------")

        def dfs(node, visited, indent=""):
            if node in visited:
                return  # Avoid revisiting nodes
            visited.add(node)

            # Print edges for this node
            for neighbor in graph.successors(node):
                print(f"{indent}{node} -> {neighbor}")
                dfs(neighbor, visited, indent + "  ")

        visited = set()

        # Start traversal from all nodes without predecessors (roots)
        for node in graph.nodes:
            if graph.in_degree(node) == 0:
                dfs(node, visited)

        # Handle disconnected components (not reachable from any "root")
        for node in graph.nodes:
            if node not in visited:
                dfs(node, visited)

        if metrics is not None:
            print("\nGraph Metrics:\n-------------")
            print(metrics)

    def export(self, output_file: str) -> str:
        """
        This method exports the DAG to a DOT file.

        Parameters:
        -----------
        dag : nx.DiGraph
            The DAG to be exported.
        output_file : str
            The path to the output DOT file.

        Returns:
        --------
        str
            The path to the output DOT file.
        """
        saved_as = utils.graph_to_dot_file(
            self.trainer[list(self.trainer.keys())[-1]].dag, output_file)

        return saved_as

    def plot(
        self,
        show_metrics: bool = False,
        show_node_fill: bool = True,
        title: str = None,
        ax: plt.Axes = None,
        figsize: Tuple[int, int] = (5, 5),
        dpi: int = 75,
        save_to_pdf: str = None,
        **kwargs
    ):
        """
        This method plots the DAG using networkx and matplotlib.

        Parameters:
        -----------
        dag : nx.DiGraph
            The DAG to be plotted.
        """
        model = self.trainer[list(self.trainer.keys())[-1]]
        if model.ref_graph is not None:
            ref_graph = model.ref_graph
        else:
            ref_graph = None
        plot.dag(
            graph=model.dag, reference=ref_graph, show_metrics=show_metrics,
            show_node_fill=show_node_fill, title=title, ax=ax,
            figsize=figsize, dpi=dpi, save_to_pdf=save_to_pdf, **kwargs)
