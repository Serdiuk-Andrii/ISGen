# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 00:34:19 2015

@author: dom
"""
import argparse
import os
import warnings
import IPython

import numpy as np
import pandas as pd
import scipy.stats
import tables

import control_calc as calc
import conv_utils
import regional_freq as freq
from ..tools.VPrint import VPrint

# Instantiate printing class globally
vprint = VPrint()


def get_regional_distributions(T, regions, contrib_h5):
    # Open allele dropping contribution results
    f = tables.open_file(contrib_h5, 'r')
    node_dict = {node.name.strip(): node
                 for node in f.list_nodes(f.root.sparse_hist)}

    # Store freqs and number of probands in each region in a dict
    expected_freqs = {}
    ninds_dict = {}

    for region in regions:
        vprint("Loading contributions to region", region)
        # data stored in sparse format as row, col, dat
        region_node = node_dict[region]
        all_contribs, ninds_region = freq.load_sparse_contribs(region_node)
        IPython.embed()
        ninds_dict[region] = ninds_region

        # Load regional contributions and parent genotypes of inds in the
        # boundary of each tree
        expected_freqs[region] = freq.get_tree_expectations(T, all_contribs)

    f.close()

    return expected_freqs, ninds_dict


def get_tree_distributions(T, all_contribs):
    """
    Returns a list of allele frequency distributions for the individuals
    in each tree in T, along with the parent genotypes of each ind in the
    boundary of the trees.
    """
    print("Calculating expectations of trees...")

    # Store ind contributions in a dict
    ind_contrib_dict = {}
    tree_expectations = np.zeros(len(T.alltrees))

    for i, tree in enumerate(T.alltrees):
        if i % 1000 == 0:
            print(f"{i}/{len(T.alltrees)}")
        # Get the boundary inds who have not been calculated already,
        # and the genotypes of their parents
        boundary_dict = T.getboundary(i)
        boundary_inds, genotypes = zip(*boundary_dict.items())
        new_inds = [ind for ind in boundary_inds if ind not in ind_contrib_dict]

        if len(new_inds) > 0:
            # Locate boundary inds in indlist
            boundary_indices = np.array([T.ind_dict[ind] for ind in new_inds])

            # Group ind contribs for the tree and normalize
            raw_contribs = all_contribs[boundary_indices].tocoo()
            contribs = freq.normalize_sparse_rows(raw_contribs)

            # Calculate expected regional allele frequencies of each tree
            boundary_weights = freq.get_genotype_weights(genotypes)
            expectations = freq.sparse_expectation(contribs, boundary_weights)

            # Store new expectations in contrib dict
            new_contribs = {ind: e for ind, e in
                            zip(new_inds, expectations)}
            ind_contrib_dict.update(new_contribs)

        # Sum expectations of boundary inds to get total for the tree
        tree_expectation = [ind_contrib_dict[ind] for ind in boundary_inds]
        tree_expectations[i] = np.sum(tree_expectation)

    return tree_expectations


def write_posteriors(outfile, posteriors, tot_liks):
    """ Writes posterior probabilities and total liks to hdf5 file """
    with tables.open_file(outfile, 'a') as f:
        title = 'Posterior probabilities of each ancestor assuming a ' + \
                'flat prior\nAnc\tProb\tLik'
        f.create_array(f.root, 'posteriors', posteriors, title=title)

        title = 'Total tree likelihoods'
        f.create_array(f.root, 'tot_liks', tot_liks, title=title)


def load_regional_contribs(contribfile, regions):
    # Load allele frequency distributions for all individuals, over
    # all probands
    region_contribs = {}

    with tables.open_file(contribfile, 'r') as f:
        for region in regions:
            global_contrib_node = f.get_node(f.root.sparse_hist, region)
            x = np.transpose(global_contrib_node[:]).reshape(3, -1)
            all_hists = scipy.sparse.csr_matrix((x[2], (x[0], x[1])))
            region_contribs[region] = all_hists

    return region_contribs


def get_boundary(T, treenum):
    """
    Returns the allele frequency distributions of individuals in the boundary
    of the tree, along with the genotypes of the parents (who are in the
    tree) of these individuals.
    """
    boundary_dict = T.getboundary(treenum)
    inds, genotypes = zip(*boundary_dict.items())

    return inds, genotypes


def get_tot_posteriors(climb_outfile, control_outfile):
    """
    Combines climbing and control likelihoods, and outputs total likelihoods
    along with posterior probabilities of each ancestor.
    """
    with tables.open_file(climb_outfile, 'r') as f:
        ancs = f.root.ancs[:].reshape(-1, 1)
        climb_liks = f.root.liks[:].reshape(-1, 1)

    with tables.open_file(control_outfile, 'r') as f:
        control_liks = f.root.control_liks[:].reshape(-1, 1)
        try:
            hap_log2lik = f.root.haplotype_liks[:].reshape(-1, 1)
            print("Found haplotype likelihoods")
        except tables.NoSuchNodeError:
            print("No haplotype likelihoods found")
            hap_log2lik = np.zeros(control_liks.shape)

    # Load likelihood data into a pandas dataframe
    lik_data = pd.DataFrame(
        np.hstack([ancs, climb_liks, control_liks, hap_log2lik]),
        columns=['anc', 'climb_log2lik', 'control_log2lik', 'hap_log2lik'])

    # Combine climbing and control likelihoods, and convert from log2 format
    lik_data['tot_log2lik'] = lik_data['climb_log2lik'] + \
                              lik_data['control_log2lik'] + \
                              lik_data['hap_log2lik']
    lik_data['tot_lik'] = lik_data['tot_log2lik'].apply(np.exp2)

    # Calculate posteriors and convert to np array for writing to h5 file
    lik_data['tot_lik'] = lik_data['tot_lik'] / len(lik_data)
    posteriors = lik_data[['anc', 'tot_lik']].groupby('anc').sum()
    posteriors['anc'] = posteriors.index
    posteriors['prob'] = posteriors['tot_lik'] / posteriors['tot_lik'].sum()

    # Change back to log2 likelihood to output alongside the posteriors
    posteriors['tot_log2lik'] = posteriors['tot_lik'].apply(np.log2)
    del posteriors['tot_lik']

    return posteriors.values, lik_data['tot_log2lik'].values


def write_regional_distributions(outfile, region, tree_dists):
    """
    Writes expected regional allele frequencies over all simulated trees,
    as well as for each individual tree.
    """
    # Filters for compression
    filters = tables.Filters(complevel=5, complib='blosc')

    # Write expected allele frequency for each tree for the region
    with tables.open_file(outfile, 'a') as f:
        # Suppress warning about naming convention
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ca = f.create_carray(f.root, region, tables.FloatAtom(),
                                 shape=tree_dists.shape, filters=filters)
        ca[:] = tree_dists


def main(args):
    # Parse CLI args
    pedfile = os.path.expanduser(args.pedfile)
    climb_outfile = os.path.expanduser(args.climb_likfile)
    contribfile = os.path.expanduser(args.contrib_file)
    outfile = os.path.expanduser(args.outfile)
    vprint.verbose = args.verbose

    vprint("Loading pedigree...")
    T = calc.Tree(pedfile, climb_outfile)
    vprint("Done.")

    # Create a dictionary of indices for all individuals, so we can locate
    # them in the contrib file
    ind_dict = dict(zip(T.inds, np.arange(len(T.inds))))

    # Load contributions of all inds in pedigree
    regions = freq.load_regions_h5(contribfile)
    region_hists = load_regional_contribs(contribfile, regions)
    for region in regions:
        C = conv_utils.BoundaryHist(region_hists[region], ind_dict)

        all_hists = []
        for tree_num in range(len(T.alltrees)):
            # We load the allele frequency distributions of the
            # boundary individuals only, as they are needed.
            inds, parent_genotypes = get_boundary(T, tree_num)

            # Get allele contribution probabilities for each ind in the boundary
            boundary_control_liks = C.get_allele_contribs(inds, parent_genotypes)
            IPython.embed()

            # Calculate likelihood of this tree having produced the observed allele frequency
            # tree_control_lik = conv_utils.sum_control_liks(boundary_control_liks, k)

            freq_dist = conv_utils.sum_control_liks(boundary_control_liks)
            all_hists.append(freq_dist)

        all_hists = np.array(all_hists)
        write_regional_distributions(outfile, region, all_hists)
        # print(freq_dist)

    IPython.embed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--verbose", action="store_true")

    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument("-p", "--pedfile", metavar='',
                               help="Pedigree file",
                               required=True)
    requiredNamed.add_argument("-l", "--climb-likfile", metavar='',
                               help="File containing results of climbing simulations",
                               required=True)
    requiredNamed.add_argument("-o", "--outfile", metavar='',
                               help="File to store allele frequency distributions" +
                                    "for each region, and each tree", required=True)
    requiredNamed.add_argument("-c", "--contrib-file", metavar='',
                               help="File containing results of allele dropping " +
                                    "simulations, for all individuals in --pedfile",
                               required=True)

    args = parser.parse_args()

    main(args)
