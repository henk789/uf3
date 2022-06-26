/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(uf3,PairUF3);
// clang-format on
#else

#ifndef LMP_PAIR_UF3_H
#define LMP_PAIR_UF3_H

#include "uf3_pair_bspline.h"
#include "uf3_triplet_bspline.h"

#include "pair.h"

#include <unordered_map>
namespace LAMMPS_NS {

class PairUF3 : public Pair {
 public:
  PairUF3(class LAMMPS *);
  ~PairUF3() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void uf3_read_pot_file(char *potf_name);
  void init_style() override;
  void init_list(int, class NeighList *) override; //needed for ptr to full neigh list
  double init_one(int, int) override; //needed for cutoff radius for neighbour list
  

 protected:
  int num_of_elements, nbody_flag, n2body_pot_files, n3body_pot_files, tot_pot_files;
  int temp_type1, temp_type2, temp_type3, temp_line_len;
  int coeff_matrix_dim1, coeff_matrix_dim2, coeff_matrix_dim3, coeff_matrix_elements_len;
  bool pot_3b;
  int ***setflag_3b;
  double **cutsq_2b, **cut_3b, cut3b_rjk, cut3b_rij, cut3b_rik;
  virtual void allocate();
  std::vector<std::vector<std::vector<double>>> n2b_knot, n2b_coeff;
  std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>> n3b_knot_matrix;
  std::vector<std::vector<std::vector<double>>> temp_3d_matrix;
  std::unordered_map<std::string, std::vector<std::vector<std::vector<double>>>> n3b_coeff_matrix;
  std::string key;
  std::vector<std::vector<uf3_pair_bspline>> UFBS2b;
  std::vector<std::vector<std::vector<uf3_triplet_bspline>>> UFBS3b;
  int *neighshort, maxshort;    // short neighbor list array for 3body interaction
  double ret_val;
  double * ret_val_deri; 
};

}    // namespace LAMMPS_NS

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

*/