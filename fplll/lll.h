/* Copyright (C) 2005-2008 Damien Stehle.
   Copyright (C) 2007 David Cade.
   Copyright (C) 2011 Xavier Pujol.

   This file is part of fplll. fplll is free software: you
   can redistribute it and/or modify it under the terms of the GNU Lesser
   General Public License as published by the Free Software Foundation,
   either version 2.1 of the License, or (at your option) any later version.

   fplll is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with fplll. If not, see <http://www.gnu.org/licenses/>. */

#ifndef FPLLL_LLL_H
#define FPLLL_LLL_H

#include "gso.h"

FPLLL_BEGIN_NAMESPACE

/* The precision of FT must be defined before creating an instance of
   LLLReduction. The matrix b can be modified between each call to lll. */

template <class ZT, class FT> class LLLReduction
{
public:
  /**
   * Constructor.
   * The precision of FT must be defined before creating an instance of the
   * class and must remain the same until the object is destroyed (or no longer
   * needed).
   */
  LLLReduction(MatGSO<ZT, FT> &m, double delta, double eta, int flags);

#ifdef FPLLL_WITH_LONG_DOUBLE
  ~LLLReduction() { LDConvHelper::free(); }
#endif

  bool lll(int kappa_min = 0, int kappa_start = 0, int kappa_end = -1);
  inline bool size_reduction(int kappa_min = 0, int kappa_end = -1);

  int status;
  int final_kappa;
  int last_early_red;
  int zeros;
  int n_swaps;

  FT delta, eta, swap_threshold;

  
private:
  bool babai(int kappa, int ncols);
  inline bool early_reduction(int start);
  inline void print_params();
  inline bool set_status(int new_status);

  MatGSO<ZT, FT> &m;

  bool enable_early_red;
  bool siegel;
  bool verbose;

  vector<FT> lovasz_tests;
  vector<FT> babai_mu;
  vector<long> babai_expo;
  ZT ztmp1;
  FT mu_m_ant, ftmp1;
};

template <class ZT, class FT> bool is_lll_reduced(MatGSO<ZT, FT> &m, double delta, double eta);

template <class ZT, class FT>
inline bool LLLReduction<ZT, FT>::size_reduction(int kappa_min, int kappa_end)
{
  if (kappa_end == -1)
    kappa_end = m.d;
  for (int k = kappa_min; k < kappa_end; k++)
  {
    if ((k > 0 && !babai(k, k)) || !m.update_gso_row(k))
      return false;
  }
  return set_status(RED_SUCCESS);
}

template <class ZT, class FT> inline bool LLLReduction<ZT, FT>::early_reduction(int start)
{
  m.lock_cols();
  if (verbose)
  {
    cerr << "Early reduction start=" << start + 1 << endl;
  }
  for (int i = start; i < m.d; i++)
  {
    if (!babai(i, start))
      return false;
  }
  m.unlock_cols();
  last_early_red = start;
  return true;
}

template <class ZT, class FT> inline void LLLReduction<ZT, FT>::print_params()
{
  cerr << "Entering LLL"
       << "\ndelta = " << delta << "\neta = " << eta << "\nprecision = " << FT::get_prec()
       << "\nexact_dot_product = " << static_cast<int>(m.enable_int_gram)
       << "\nrow_expo = " << static_cast<int>(m.enable_row_expo)
       << "\nearly_red = " << static_cast<int>(enable_early_red)
       << "\nsiegel_cond = " << static_cast<int>(siegel)
       << "\nlong_in_babai = " << static_cast<int>(m.row_op_force_long) << endl;
}

template <class ZT, class FT> inline bool LLLReduction<ZT, FT>::set_status(int new_status)
{
  status = new_status;
  if (verbose)
  {
    if (status == RED_SUCCESS)
      cerr << "End of LLL: success" << endl;
    else
      cerr << "End of LLL: failure: " << RED_STATUS_STR[status] << endl;
  }
  return status == RED_SUCCESS;
}

FPLLL_END_NAMESPACE

#endif
