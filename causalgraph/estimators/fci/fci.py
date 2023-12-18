# pylint: disable=E1101:no-member
# pylint: disable=W0201:attribute-defined-outside-init, W0511:fixme
# pylint: disable=W0106:expression-not-assigned
# pylint: disable=C0103:invalid-name, C0116:missing-function-docstring
# pylint: disable=R0913:too-many-arguments, R0902:too-many-instance-attributes
# pylint: disable=R0914:too-many-locals, R0915:too-many-statements
# pylint: disable=R1702:too-many-branches

import multiprocessing as mp
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

from causallearn.search.ConstraintBased.FCI import fci as cl_fci
from causallearn.utils.cit import kci

import pandas as pd
from networkx import Graph
from sklearn.discriminant_analysis import StandardScaler
from tqdm.auto import tqdm

from causalgraph.estimators.fci.initialization import (dsep_set_from_csv,
                                                       save_graph, save_sepset)
from causalgraph.common.utils import graph_from_adjacency_file, graph_from_dot_file
from causalgraph.estimators.fci.colliders import (get_dsep_combs,
                                                  get_neighbors, init_pag,
                                                  orientEdges)
from causalgraph.estimators.fci.debug import DebugFCI
from causalgraph.estimators.fci.graph_learner import GraphLearner
from causalgraph.estimators.fci.pag import PAG
from causalgraph.independence.hsic import HSIC

#
# TODO: No control over max_samples to be used in HSIC `check_independence`
#


class FCI(GraphLearner):
    """
    A graph learner which implements the FCI algorithm
    """

    pag = None
    dag = None

    def __init__(
            self,
            data_file: str,
            output_path: Union[Path, str],
            logger: Optional[Callable] = None,
            indep_test: Callable = HSIC,
            load_base_skeleton: bool = False,
            load_final_skeleton: bool = False,
            save_intermediate: bool = False,
            base_skeleton: str = None,
            base_sepset: str = None,
            final_skeleton: str = None,
            final_sepset: str = None,
            njobs: int = 1,
            verbose: bool = False,
            *args,
            **kwargs):
        """
        Initialize the FCI algorithm creating an FCI learner.

        Parameters
        ----------
        data (pd.DataFrame): data to be used for learning the causal graph
        data_file (str): name of the file containing the data
        output_path (Path): path to the directory where the output files will be saved
        logger (logging.Logger): logger object
        indep_test (Callable): independence test to be used for checking independence
        load_base_skeleton (bool): if True, load the base skeleton from the files specified
            in the YAML file for "base_skeleton" and "base_sepset"
        load_final_skeleton (bool): if True, load the final skeleton from the files specified
            in the YAML file for "final_skeleton" and "final_sepset"
        save_intermediate (bool): if True, save the intermediate results
        base_skeleton (str): name of the file containing the base skeleton
        base_sepset (str): name of the file containing the base sepset
        final_skeleton (str): name of the file containing the final skeleton
        final_sepset (str): name of the file containing the final sepset
        njobs (int): number of jobs to be used for parallel processing. If 1, run in
            sequential mode.
        verbose (bool): if True, print the progress of the algorithm

        Returns
        -------
        FCI object.
        """

        super().__init__(logger=logger,  # data=data,
                         data_file=data_file,
                         indep_test=indep_test,
                         parallel=njobs > 1,
                         verbose=verbose,
                         *args, **kwargs)
        self.data_file = data_file
        self.indep_test = indep_test
        self.load_final_skeleton = load_final_skeleton
        self.save_intermediate = save_intermediate
        self.output_path = output_path
        self.final_skeleton = final_skeleton
        self.final_sepset = final_sepset
        self.njobs = njobs
        self.verbose = verbose
        self.load_base_skeleton = load_base_skeleton
        self.base_skeleton = base_skeleton
        self.base_sepset = base_sepset
        self.log = logger

        if verbose:
            self.debug = DebugFCI(self.verbose)

    def fit(self, data: pd.DataFrame):
        """
        function to learn a causal network from data

        Returns
        -------
        PAG
            causal network learned from data
        """
        self.data = data
        super()._init_data(self.data)
        if self.verbose:
            print("Getting Skeleton of graph...")
        start_time = time.time()
        skeleton, sepset = self.learn_or_load_final_skeleton()
        if self.verbose:
            print("Orienting Edges...")
        self.pag = orientEdges(skeleton, sepset, data_file=self.data_file,
                               output_path=self.output_path, log=self.log,
                               verbose=self.verbose, debug=self.debug)
        self.dag = self.pag

        if self.verbose:
            print("Learning complete")
        self.debug.bm(f"Total time: {time.time() - start_time:.1f} secs.")

        return self

    def learn_or_load_final_skeleton(self) -> Tuple[Graph, Dict]:
        """
        Learn a new Skeleton (2nd stage) or load an existing one if the files specified
        as parameters in the YAML file for "final_skeleton" and "final_sepset" exist.

        :return: graph, dict containing the skeleton and separation sets.
        """
        if self.load_final_skeleton:
            sk_file = Path(self.output_path, self.final_skeleton)
            ss_file = Path(self.output_path, self.final_sepset)
            if os.path.exists(sk_file) and os.path.exists(ss_file):
                skeleton = graph_from_adjacency_file(sk_file)
                sepset = dsep_set_from_csv(ss_file)
                if self.log:
                    self.log.info(f"Read {sk_file}")
                    self.log.info(f"Read {ss_file}")
            else:
                raise FileNotFoundError("Cannot find FINAL skeleton or sepset")
        else:
            now = time.time()
            if self.njobs > 1:
                skeleton, sepset = self.parallel_learn_skeleton()
            else:
                skeleton, sepset = self.learn_skeleton()
            self.debug.bm(
                f"2nd Stage Skeleton time: {time.time() - now:.1f} secs.")
            if self.save_intermediate:
                save_graph(skeleton, prefix="final_skeleton_FCI",
                           data_file=self.data_file,
                           output_path=self.output_path, log=self.log)
                save_sepset(sepset, prefix="final_sepset_FCI",
                            data_file=self.data_file, output_path=self.output_path,
                            log=self.log)
        return skeleton, sepset

    def learn_skeleton(self):
        """
        A  function to build the skeleton of a causal graph from data
        :returns: PDAG(The skeleton of the causal network) and a dict containg
        separation sets of all pairs of nodes
        """
        if self.log:
            self.log.info("Running in sequential mode")

        # Learn the base skeleton first.
        skeleton, sepSet = self.learn_or_load_base_skeleton()
        if self.verbose:
            for edge in skeleton.edges():
                print(edge)

        # Initialization
        pag_actions = defaultdict(list)
        pag, dseps = init_pag(skeleton, sepSet, self.verbose, self.debug)
        if self.verbose:
            print("Finding colliders...", flush=True)
        pbar = tqdm(pag, disable=self.verbose, leave=False)
        pbar.set_description(f"Learn Skeleton")
        for lx, x in enumerate(pag):
            # if x not in ["F", "J"]:
            #     continue
            pbar.update()
            neighbors = get_neighbors(x, pag)
            self.debug.neighbors(x, lx, neighbors, pag)
            if not len(neighbors):
                continue
            for ny, y in enumerate(neighbors):
                tup = self.find_colliders(x, y, pag, ny, dseps, sepSet,
                                          neighbors)
                if tup:
                    self.update_actions(pag_actions, tup)
            self.debug.stack(pag_actions)
            pbar.refresh()
        pbar.close()
        return pag, sepSet

    def parallel_learn_skeleton(self):
        """
        PARALLEL version of Learn Skeleton
        :return: tuple(PAG, sepSet)
        """
        if self.log:
            self.log.info(f"Running in parallel {self.njobs}")
        # Learn the base skeleton first.
        skeleton, sepSet = super().learn_or_load_base_skeleton()

        # Initialization
        now = time.time()
        pag_actions = defaultdict(list)
        pag, dseps = init_pag(skeleton, sepSet, self.verbose, self.debug)
        results = []
        pbar = None

        def update_PAG(tup):
            nonlocal results, pag, dseps, pag_actions, pbar
            if tup is None:
                results.append(False)
                pbar.update(1)
                return
            orig_node, dest_node, sep_set = tup[0], tup[1], tup[2]
            self.update_actions(pag_actions, (orig_node, dest_node, sep_set))
            if pag.has_edge(orig_node, dest_node):
                pag.remove_edge(orig_node, dest_node)
            dseps[(orig_node, dest_node)] = sep_set
            dseps[(dest_node, orig_node)] = sep_set
            results.append(True)
            pbar.update(1)

        if self.verbose:
            print("Finding colliders...", flush=True)
        pool = mp.Pool(self.njobs)
        pbar = tqdm(total=len(pag), disable=self.verbose, leave=False)
        pbar.set_description("Learn skeleton")
        results = []
        for lx, x in enumerate(pag):
            # if x not in ["F", "J"]:
            #     continue
            pbar.update()
            pool.apply_async(
                self.explore_neighbours,
                args=(x, lx, pag, dseps, sepSet),
                callback=update_PAG)
            pbar.refresh()
        pool.close()
        # Wait until everyone is finished...
        pool.join()
        pbar.close()
        self.debug.stack(pag_actions)
        self.debug.bm(f"2nd stage skeleton time: {time.time() - now:.1f} secs")
        return pag, sepSet

    def explore_neighbours(self, x, id_x, pag, dseps, sepSet):
        # self.oo(f"–––––– {mp.current_process().name} ––––––")
        neighbors = get_neighbors(x, pag)
        self.debug.neighbors(x, id_x, neighbors, pag)
        if len(neighbors) == 0:
            return None
        for ny, y in enumerate(neighbors):
            tup = self.find_colliders(x, y, pag, ny, dseps, sepSet, neighbors)
            if tup:
                return tup
        return None

    def find_colliders(self, x, y, pag, ny, dseps, sepSet, neighbors):
        self.debug.y(x, y, ny, dseps, neighbors)
        i = 0
        while i < len(dseps[x]) + 1:
            tup = self.test_independence(x, y, i, dseps, sepSet, pag)
            if tup:
                self.debug.interrupt()
                return tup
            i += 1
        return None

    def test_independence(self, x: str, y: str, comb_len: int, dseps, sepSet,
                          pag: PAG) -> Optional[Tuple[str, str, list]]:
        dsep_combinations = get_dsep_combs(dseps, x, y, comb_len)
        if len(dsep_combinations) == 0:
            # self.debug.empty_set(comb_len, dseps, x, y)
            return None

        # Walk through all combinations formed for this d-sep.
        self.debug.d_seps(comb_len, dseps, dsep_combinations, x, y)
        for idx, dsep in enumerate(dsep_combinations):
            dsep = list(dsep)
            if len(dsep) < 1:
                continue
            if self.is_independent(x, y, dsep, idx, len(dsep_combinations)):
                self.update_graph(x, y, dsep, pag, sepSet)
                self.debug.interrupt()
                return x, y, dsep
        return None


def main(dataset_name,
         input_path="/Users/renero/phd/data/RC3/",
         output_path="/Users/renero/phd/output/RC3/",
         save=False,
         **kwargs):
    """
    Create a call to FCI with a sample dataset.
    """
    ref_graph = graph_from_dot_file(f"{input_path}{dataset_name}.dot")
    data = pd.read_csv(f"{input_path}{dataset_name}.csv")
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    train = data.sample(frac=0.8, random_state=42)
    test = data.drop(train.index)

    njobs = kwargs.get("njobs", 1)
    indep_test = kwargs.get("indep_test", HSIC)
    load_base_skeleton = kwargs.get("load_base_skeleton", False)
    load_final_skeleton = kwargs.get("load_final_skeleton", False)
    save_intermediate = kwargs.get("save_intermediate", False)
    base_skeleton = kwargs.get("base_skeleton", None)
    base_sepset = kwargs.get("base_sepset", None)
    final_skeleton = kwargs.get("final_skeleton", None)
    final_sepset = kwargs.get("final_sepset", None)
    fci = FCI(
        # data=data,
        data_file=dataset_name,
        output_path=output_path,
        njobs=njobs,
        indep_test=indep_test,
        load_base_skeleton=load_base_skeleton,
        load_final_skeleton=load_final_skeleton,
        save_intermediate=save_intermediate,
        base_skeleton=base_skeleton,
        base_sepset=base_sepset,
        final_skeleton=final_skeleton,
        final_sepset=final_sepset,
    )
    fci.fit(data)

    # dag, edges = cl_fci(
    #     data.values,
    #     independence_test_method=kci,
    #     alpha=0.05,
    #     depth=-1,
    #     max_path_length=3,
    #     verbose=False,
    #     background_knowledge=None,
    #     cache_variables_map=None)

    for edge in fci.dag.edges():
        print(edge)

    # if save:
    #     where_to = utils.save_experiment(rex.name, output_path, rex)
    #     print(f"Saved '{rex.name}' to '{where_to}'")


# Create a call to FCI with a sample dataset.
if __name__ == "__main__":
    main("rex_generated_linear_0", njobs=1)
