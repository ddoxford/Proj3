#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 14:34:19 2021

@author: kendrick
"""

import numpy as np

# compute unknown displacements 
def ComputeDisplacements(K, F, n_unknowns):
    # extract submatrix of unknowns
    K11 = K[0:n_unknowns,0:n_unknowns]
    F1 = F[0:n_unknowns]
    
    d = np.linalg.solve(K11,F1)
    
    return d

# postprocess the forces at known displacement nodes
def PostprocessReactions(K, d, F, n_unknowns, nodes):
    # These are computed net forces and do not
    # take into account external loads applied
    # at these nodes
    F = np.matmul(K[n_unknowns:,0:n_unknowns], d)
    
    # Postprocess the reactions
    for node in nodes:
        if node.xidx >= n_unknowns:
            node.AddReactionXForce(F[node.xidx-n_unknowns][0] - node.xforce_external)
        if node.yidx >= n_unknowns:
            node.AddReactionYForce(F[node.yidx-n_unknowns][0] - node.yforce_external)
        
    return F

# determine internal member loads
def ComputeMemberForces(bars):
    # Compute member forces for all bars using equation 14-23:
    #   q_F = (AE/L) * [-lambda_x, -lambda_y, lambda_x, lambda_y] . [D_Nx, D_Ny, D_Fx, D_Fy]
    # Units: E (ksi), A (in^2), L (ft), displacements (ft)  ->  q_F in kips
    # Positive q_F = tension, negative q_F = compression
    for bar in bars:
        # extract bar properties
        E = bar.E
        A = bar.A
        L = bar.Length()
        [lambdax, lambday] = bar.LambdaTerms()

        # extract the near and far nodes of the bar
        near_node = bar.init_node
        far_node = bar.end_node

        # extract the global displacements of both nodes
        disp = np.array([near_node.xdisp,
                         near_node.ydisp,
                         far_node.xdisp,
                         far_node.ydisp])

        # transformation row vector from Eq. 14-23
        transform = np.array([-lambdax, -lambday, lambdax, lambday])

        # compute and store the axial (internal) force in the bar
        q_F = (A*E/L) * np.dot(transform, disp)
        bar.axial_load = q_F

# compute the normal stresses
def ComputeNormalStresses(bars):
    # Compute normal stress for all bars: sigma = P / A
    # Units: axial load (kips) / A (in^2) -> stress in ksi
    for bar in bars:
        P = bar.axial_load
        A = bar.A
        bar.normal_stress = P / A

# compute the critical buckling load of a member
def ComputeBucklingLoad(bars):
    # Compute critical buckling load for all bars: Pcr = pi^2 E I / (K L)^2
    # Units: E (ksi), I (in^4), L converted from ft to in -> Pcr in kips
    K = 1.0  # effective length factor for pin-connected truss members
    for bar in bars:
        E = bar.E
        # A truss member is unbraced along its length in both directions,
        # so it buckles about the axis with the smaller second moment of
        # inertia (the weak axis for a W shape)
        I = min(bar.GetStrongSecondMoment(), bar.GetWeakSecondMoment())
        L = bar.Length() * 12  # ft -> in
        bar.buckling_load = (np.pi**2 * E * I) / (K * L)**2