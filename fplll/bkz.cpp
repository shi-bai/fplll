/* Copyright (C) 2011 Xavier Pujol
   (C) 2014-2016 Martin R. Albrecht
   (C) 2016 Michael Walter

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

#include <iomanip>

#include "bkz.h"
#include "bkz_param.h"
#include "enum/enumerate.h"
#include "util.h"
#include "wrapper.h"
#include <iomanip>

FPLLL_BEGIN_NAMESPACE

template <class FT>
BKZReduction<FT>::BKZReduction(MatGSO<Integer, FT> &m, LLLReduction<Integer, FT> &lll_obj,
                               const BKZParam &param)
    : status(RED_SUCCESS), nodes(0), param(param), m(m), lll_obj(lll_obj), algorithm(NULL),
      cputime_start(0)
{
  for (num_rows = m.d; num_rows > 0 && m.b[num_rows - 1].is_zero(); num_rows--)
  {
  }
  this->delta = param.delta;
}

template <class FT> BKZReduction<FT>::~BKZReduction() {}

template <class FT> void BKZReduction<FT>::rerandomize_block(int min_row, int max_row, int density)
{
  if (max_row - min_row < 2)
    return;

  // 1. permute rows
  size_t niter = 4 * (max_row - min_row);  // some guestimate

  for (size_t i = 0; i < niter; ++i)
  {
    size_t a = gmp_urandomm_ui(RandGen::get_gmp_state(), max_row - min_row - 1) + min_row;
    size_t b = a;
    while (b == a)
    {
      b = gmp_urandomm_ui(RandGen::get_gmp_state(), max_row - min_row - 1) + min_row;
    }
    m.move_row(b, a);
  }

  // 2. triangular transformation matrix with coefficients in -1,0,1
  m.row_op_begin(min_row, max_row);
  FT x;
  for (long a = min_row; a < max_row - 2; ++a)
  {
    for (long i = 0; i < density; i++)
    {
      size_t b = gmp_urandomm_ui(RandGen::get_gmp_state(), max_row - (a + 1) - 1) + a + 1;
      if (gmp_urandomm_ui(RandGen::get_gmp_state(), 2))
        m.row_add(a, b);
      else
        m.row_sub(a, b);
    }
  }
  m.row_op_end(min_row, max_row);

  return;
}

template <class FT>
const Pruning &BKZReduction<FT>::get_pruning(int kappa, int block_size, const BKZParam &par) const
{

  FPLLL_DEBUG_CHECK(param.strategies.size() > block_size);

  Strategy &strat = par.strategies[block_size];

  long max_dist_expo;
  FT max_dist    = m.get_r_exp(kappa, kappa, max_dist_expo);
  FT gh_max_dist = max_dist;
  FT root_det    = m.get_root_det(kappa, kappa + block_size);
  gaussian_heuristic(gh_max_dist, max_dist_expo, block_size, root_det, 1.0);
  return strat.get_pruning(max_dist.get_d() * pow(2, max_dist_expo),
                           gh_max_dist.get_d() * pow(2, max_dist_expo));
}

template <class FT>
bool BKZReduction<FT>::svp_preprocessing(int kappa, int block_size, const BKZParam &param)
{
  bool clean = true;

  FPLLL_DEBUG_CHECK(param.strategies.size() > block_size);

  int lll_start = (param.flags & BKZ_BOUNDED_LLL) ? kappa : 0;
  double start_time = cputime();/* TIMING */
  if (!lll_obj.lll(lll_start, lll_start, kappa + block_size))
  {
    throw std::runtime_error(RED_STATUS_STR[lll_obj.status]);
  }
  cputime_others += (cputime() - start_time);/* TIMING */
  cputime_others_lll += (cputime() - start_time);/* TIMING */
  cputime_others_lll_svppre += (cputime() - start_time);/* TIMING */
  if (lll_obj.n_swaps > 0)
    clean = false;

  auto &preproc = param.strategies[block_size].preprocessing_block_sizes;
  for (auto it = preproc.begin(); it != preproc.end(); ++it)
  {
    int dummy_kappa_max = num_rows;
    BKZParam prepar     = BKZParam(*it, param.strategies, LLL_DEF_DELTA, BKZ_GH_BND);
    clean &= tour(0, dummy_kappa_max, prepar, kappa, kappa + block_size);
  }

  return clean;
}

template <class FT>
bool BKZReduction<FT>::svp_postprocessing(int kappa, int block_size, const vector<FT> &solution)
{
  // Is it already in the basis ?
  double start_time = 0.0;/* TIMING */
  int nz_vectors = 0, i_vector = -1;
  for (int i = 0; i < block_size; i++)
  {
    if (!solution[i].is_zero())
    {
      nz_vectors++;
      if (i_vector == -1 && fabs(solution[i].get_d()) == 1)
        i_vector = i;
    }
  }
  FPLLL_DEBUG_CHECK(nz_vectors > 0);

  if (nz_vectors == 1)
  {
    // Yes, it is another vector
    FPLLL_DEBUG_CHECK(i_vector != -1 && i_vector != 0);
    m.move_row(kappa + i_vector, kappa);
    if (!lll_obj.size_reduction(kappa, kappa + i_vector + 1))
      throw lll_obj.status;
  }
  else
  {
    int found_one = 0;
    i_vector = block_size;
    while ((found_one == 0)&&(i_vector>0)){
      i_vector--;
      if (fabs(solution[i_vector].get_d()) == 1)
        found_one = 1;
    }

    if (found_one == 1)
    {
      start_time = cputime();/* TIMING */
      // No, general case
      int d = m.d;
      m.create_row();
      m.row_op_begin(d, d + 1);
      for (int i = 0; i < block_size; i++)
      {
        m.row_addmul(d, kappa + i, solution[i]);
      }
      m.row_op_end(d, d + 1);
      m.move_row(d, kappa);
      m.move_row(kappa+i_vector+1,d);
      m.remove_last_row();
      if (!lll_obj.size_reduction(kappa, kappa + block_size))
        throw lll_obj.status;
      FPLLL_DEBUG_CHECK(m.b[kappa + block_size].is_zero());
      cputime_others += (cputime() - start_time);/* TIMING */
      cputime_others_lll += (cputime() - start_time);/* TIMING */
      cputime_others_lll_svppost += (cputime() - start_time);/* TIMING */
    }
    else
    {
      start_time = cputime();/* TIMING */
      // No, general case
      int d = m.d;
      m.create_row();
      m.row_op_begin(d, d + 1);
      for (int i = 0; i < block_size; i++)
      {
        m.row_addmul(d, kappa + i, solution[i]);
      }
      m.row_op_end(d, d + 1);
      m.move_row(d, kappa);
      if (!lll_obj.lll(kappa, kappa, kappa + block_size + 1))
        throw lll_obj.status;
      FPLLL_DEBUG_CHECK(m.b[kappa + block_size].is_zero());
      m.move_row(kappa + block_size, d);
      m.remove_last_row();
      cputime_others += (cputime() - start_time);/* TIMING */
      cputime_others_lll += (cputime() - start_time);/* TIMING */
      cputime_others_lll_svppost += (cputime() - start_time);/* TIMING */
    }
  }
  
  return false;
}

template <class FT>
bool BKZReduction<FT>::dsvp_postprocessing(int kappa, int block_size, const vector<FT> &solution)
{
  double start_time;/* TIMING */
  vector<FT> x = solution;

  int d = block_size;
  m.row_op_begin(kappa, kappa + d);
  // don't want to deal with negativ coefficients
  for (int i = 0; i < d; i++)
  {
    if (x[i] < 0)
    {
      x[i].neg(x[i]);
      for (int j = 0; j < m.b.get_cols(); j++)
      {
        m.b[i + kappa][j].neg(m.b[i + kappa][j]);
      }
    }
  }
  start_time = cputime();/* TIMING */
  // tree based gcd computation on x, performing operations also on b
  int off = 1;
  int k;
  while (off < d)
  {
    k = d - 1;
    while (k - off >= 0)
    {
      if (!(x[k].is_zero() && x[k - off].is_zero()))
      {
        if (x[k] < x[k - off])
        {
          x[k].swap(x[k - off]);
          m.b.swap_rows(kappa + k, kappa + k - off);
        }

        while (!x[k - off].is_zero())
        {
          while (x[k - off] <= x[k])
          {
            x[k] = x[k] - x[k - off];
            m.b[kappa + k].sub(m.b[kappa + k - off]);
          }

          x[k].swap(x[k - off]);
          m.b.swap_rows(kappa + k, kappa + k - off);
        }
      }
      k -= 2 * off;
    }
    off *= 2;
  }
  m.row_op_end(kappa, kappa + d);
  if (!lll_obj.lll(kappa, kappa, kappa + d))
  {
    return set_status(lll_obj.status);
  }
  cputime_others += (cputime() - start_time);/* TIMING */
  cputime_others_lll += (cputime() - start_time);/* TIMING */
  cputime_others_lll_svppost += (cputime() - start_time);/* TIMING */
  return false;
}

template <class FT>
bool BKZReduction<FT>::svp_reduction(int kappa, int block_size, const BKZParam &par, bool dual)
{

  int first = dual ? kappa + block_size - 1 : kappa;

  // ensure we are computing something sensible.
  // note that the size reduction here is required, since
  // we're calling this function on unreduced blocks at times
  // (e.g. in bkz, after the previous block was reduced by a vector
  // already in the basis). if size reduction is not called,
  // old_first might be incorrect (e.g. close to 0) and the function
  // will return an incorrect clean flag
  double start_time = cputime();/* TIMING */
  if (!lll_obj.size_reduction(0, first + 1))
  {
    throw std::runtime_error(RED_STATUS_STR[lll_obj.status]);
  }
  cputime_others += (cputime() - start_time);/* TIMING */
  cputime_others_lll += (cputime() - start_time);/* TIMING */
  cputime_others_lll_svpred1 += (cputime() - start_time);/* TIMING */

  FT old_first;
  long old_first_expo;
  old_first = FT(m.get_r_exp(first, first, old_first_expo));

  bool rerandomize             = false;
  double remaining_probability = 1.0;

  while (remaining_probability > 1. - par.min_success_probability)
  {
    start_time = cputime();/* TIMING */
    if (rerandomize)
    {
      rerandomize_block(kappa + 1, kappa + block_size, par.rerandomization_density);
    }
    cputime_others += (cputime() - start_time);/* TIMING */
    cputime_others_random += (cputime() - start_time);/* TIMING */
    svp_preprocessing(kappa, block_size, par);/* TIMING INSIDE */

    long max_dist_expo;
    FT max_dist = m.get_r_exp(first, first, max_dist_expo);
    if (dual)
    {
      max_dist.pow_si(max_dist, -1, GMP_RNDU);
      max_dist_expo *= -1;
    }
    max_dist *= delta;

    if ((par.flags & BKZ_GH_BND) && block_size > 30)
    {
      FT root_det = m.get_root_det(kappa, kappa + block_size);
      gaussian_heuristic(max_dist, max_dist_expo, block_size, root_det, par.gh_factor);
    }

    start_time = cputime();
    const Pruning &pruning = get_pruning(kappa, block_size, par);

    vector<FT> &sol_coord = evaluator.sol_coord;
    sol_coord.clear();
    Enumeration<FT> enum_obj(m, evaluator);
    enum_obj.enumerate(kappa, kappa + block_size, max_dist, max_dist_expo, vector<FT>(),
                       vector<enumxt>(), pruning.coefficients, dual);
    nodes += enum_obj.get_nodes();
    cputime_svp += (cputime() - start_time);

    if (!sol_coord.empty())
    {
      if (dual)
        dsvp_postprocessing(kappa, block_size, sol_coord);/* TIMING INSIDE */
      else
        svp_postprocessing(kappa, block_size, sol_coord);/* TIMING INSIDE */

      rerandomize = false;
    }
    else
    {
      rerandomize = true;
    }
    remaining_probability *= (1 - pruning.probability);
  }
  start_time = cputime();
  if (!lll_obj.size_reduction(0, first + 1))
  {
    throw std::runtime_error(RED_STATUS_STR[lll_obj.status]);
  }
  cputime_others += (cputime() - start_time);
  cputime_others_lll += (cputime() - start_time);
  cputime_others_lll_svpred1 += (cputime() - start_time);/* TIMING */
  long new_first_expo;
  FT new_first = m.get_r_exp(first, first, new_first_expo);
  new_first.mul_2si(new_first, new_first_expo - old_first_expo);
  return (dual) ? (old_first >= new_first) : (old_first <= new_first);
}

template <class FT>
bool BKZReduction<FT>::tour(const int loop, int &kappa_max, const BKZParam &par, int min_row,
                            int max_row)
{
  bool clean = true;
  clean &= trunc_tour(kappa_max, par, min_row, max_row);
  clean &= hkz(kappa_max, par, max(max_row - par.block_size, 0), max_row);

  if (par.flags & BKZ_VERBOSE)
  {
    print_tour(loop, min_row, max_row);
  }

  if (par.flags & BKZ_DUMP_GSO)
  {
    std::ostringstream prefix;
    prefix << "End of BKZ loop " << std::setw(4) << loop;
    prefix << " (" << std::fixed << std::setw(9) << std::setprecision(3)
           << (cputime() - cputime_start) * 0.001 << "s)";
    dump_gso(par.dump_gso_filename, prefix.str());
  }

  return clean;
}

template <class FT>
bool BKZReduction<FT>::trunc_tour(int &kappa_max, const BKZParam &par, int min_row, int max_row)
{
  bool clean     = true;
  int block_size = par.block_size;
  for (int kappa = min_row; kappa < max_row - block_size; ++kappa)
  {
    clean &= svp_reduction(kappa, block_size, par);
    /*
    if ((par.flags & BKZ_VERBOSE) && kappa_max < kappa && clean)
    {
      cerr << "Block [1-" << setw(4) << kappa + 1 << "] BKZ-" << setw(0) << par.block_size
           << " reduced for the first time" << endl;
      kappa_max = kappa;
      }*/
    print_after_svp (0, max_row, block_size);
  }

  return clean;
}

template <class FT>
bool BKZReduction<FT>::trunc_dtour(const BKZParam &par, int min_row, int max_row)
{
  bool clean     = true;
  int block_size = par.block_size;

  for (int kappa = max_row - block_size; kappa > min_row; --kappa)
  {
    clean &= svp_reduction(kappa, block_size, par, true);
  }

  return clean;
}

template <class FT>
bool BKZReduction<FT>::hkz(int &kappa_max, const BKZParam &param, int min_row, int max_row)
{
  bool clean = true;
  for (int kappa = min_row; kappa < max_row - 1; ++kappa)
  {
    int block_size = max_row - kappa;
    clean &= svp_reduction(kappa, block_size, param);
    /*
    if ((param.flags & BKZ_VERBOSE) && kappa_max < kappa && clean)
    {
      cerr << "Block [1-" << setw(4) << kappa + 1 << "] BKZ-" << setw(0) << param.block_size
           << " reduced for the first time" << endl;
      kappa_max = kappa;
      }*/
    print_after_svp (0, max_row, block_size);
  }

  return clean;
}

template <class FT>
bool BKZReduction<FT>::sd_tour(const int loop, const BKZParam &par, int min_row, int max_row)
{
  int dummy_kappa_max = num_rows;
  bool clean          = true;
  clean &= trunc_dtour(par, min_row, max_row);
  clean &= trunc_tour(dummy_kappa_max, par, min_row, max_row);

  if (par.flags & BKZ_VERBOSE)
  {
    print_tour(loop, min_row, max_row);
  }

  if (par.flags & BKZ_DUMP_GSO)
  {
    std::ostringstream prefix;
    prefix << "End of SD-BKZ loop " << std::setw(4) << loop;
    prefix << " (" << std::fixed << std::setw(9) << std::setprecision(3)
           << (cputime() - cputime_start) * 0.001 << "s)";
    dump_gso(par.dump_gso_filename, prefix.str());
  }

  return clean;
}

template <class FT>
bool BKZReduction<FT>::slide_tour(const int loop, const BKZParam &par, int min_row, int max_row)
{
  int p = (max_row - min_row) / par.block_size;
  if ((max_row - min_row) % par.block_size)
    ++p;
  bool clean;  // this clean variable is only for the inner loop of slide reduction
  do
  {
    clean = true;
    // SVP reduction takes care of the LLL reduction as long as BKZ_BOUNDED_LLL is off
    for (int i = 0; i < p; ++i)
    {
      int kappa      = min_row + i * par.block_size;
      int block_size = min(max_row - kappa, par.block_size);
      clean &= svp_reduction(kappa, block_size, par);
    }
  } while (!clean);

  for (int i = 0; i < p - 1; ++i)
  {
    int kappa = min_row + i * par.block_size + 1;
    svp_reduction(kappa, par.block_size, par, true);
  }

  FT new_potential = m.get_slide_potential(min_row, max_row, par.block_size);

  if (par.flags & BKZ_VERBOSE)
  {
    print_tour(loop, min_row, max_row);
  }

  if (par.flags & BKZ_DUMP_GSO)
  {
    std::ostringstream prefix;
    prefix << "End of SLD loop " << std::setw(4) << loop;
    prefix << " (" << std::fixed << std::setw(9) << std::setprecision(3)
           << (cputime() - cputime_start) * 0.001 << "s)";
    dump_gso(par.dump_gso_filename, prefix.str());
  }

  if (new_potential >= sld_potential)
    return true;

  sld_potential = new_potential;
  return false;
}

template <class FT> bool BKZReduction<FT>::bkz()
{
  int flags        = param.flags;
  int final_status = RED_SUCCESS;
  nodes            = 0;
  bool sd          = (flags & BKZ_SD_VARIANT);
  bool sld         = (flags & BKZ_SLD_RED);
  algorithm        = sd ? "SD-BKZ" : sld ? "SLD" : "BKZ";

  /* changed from main */
  num_svp  = 0;
  num_dsvp            = 0;
  svp_bs_count.resize(param.block_size);
  input_block_size = param.block_size;
  best_so_far = std::numeric_limits<double>::max();

  
  if (sd && sld)
  {
    throw std::runtime_error("Invalid flags: SD-BKZ and Slide reduction are mutually exclusive!");
  }

  if (flags & BKZ_DUMP_GSO)
  {
    std::ostringstream prefix;
    prefix << "Input";
    dump_gso(param.dump_gso_filename, prefix.str(), false);
  }

  if (param.block_size < 2)
    return set_status(RED_SUCCESS);

  int i = 0;

  BKZAutoAbort<FT> auto_abort(m, num_rows);

  if (sd && !(flags & (BKZ_MAX_LOOPS | BKZ_MAX_TIME | BKZ_AUTO_ABORT)))
  {
    cerr << "Warning: SD Variant of BKZ requires explicit termination condition. Turning auto "
            "abort on!"
         << endl;
    flags |= BKZ_AUTO_ABORT;
  }

  if (flags & BKZ_VERBOSE)
  {
    cerr << "Entering " << algorithm << ":" << endl;
    print_params(param, cerr);
    cerr << endl;
  }
  cputime_start = cputime();
  cputime_svp = 0.0;
  cputime_others = 0.0;  
  cputime_others_lll = 0.0;
  cputime_others_lll_svppre = 0.0;
  cputime_others_lll_svppost = 0.0;  
  cputime_others_lll_svpred1 = 0.0;
  cputime_others_random = 0.0;  

  m.discover_all_rows();

  if (sld)
  {
    m.update_gso();
    sld_potential = m.get_slide_potential(0, num_rows, param.block_size);
  }

  // the following is necessary, since sd-bkz starts with a dual tour and
  // svp_reduction calls size_reduction, which needs to be preceeded by a
  // call to lll lower blocks to avoid seg faults
  if (sd)
    lll_obj.lll(0, 0, num_rows);

  int kappa_max;
  bool clean = true;
  for (i = 0;; ++i)
  {
    if ((flags & BKZ_MAX_LOOPS) && i >= param.max_loops)
    {
      final_status = RED_BKZ_LOOPS_LIMIT;
      break;
    }
    if ((flags & BKZ_MAX_TIME) && (cputime() - cputime_start) * 0.001 >= param.max_time)
    {
      final_status = RED_BKZ_TIME_LIMIT;
      break;
    }
    if ((flags & BKZ_AUTO_ABORT) &&
        auto_abort.test_abort(param.auto_abort_scale, param.auto_abort_max_no_dec))
    {
      break;
    }

    try
    {
      if (sd)
      {
        clean = sd_tour(i, param, 0, num_rows);
      }
      else if (sld)
      {
        clean = slide_tour(i, param, 0, num_rows);
      }
      else
      {
        clean = tour(i, kappa_max, param, 0, num_rows);
      }
    }
    catch (RedStatus &e)
    {
      return set_status(e);
    }

    if (clean || param.block_size >= num_rows)
      break;
  }

  int dummy_kappa_max = num_rows;
  if (sd)
  {
    try
    {
      hkz(dummy_kappa_max, param, num_rows - param.block_size, num_rows);
      print_tour(i, 0, num_rows);
    }
    catch (RedStatus &e)
    {
      return set_status(e);
    }
  }
  if (sld)
  {
    try
    {
      int p = num_rows / param.block_size;
      if (num_rows % param.block_size)
        ++p;
      for (int j = 0; j < p; ++j)
      {
        int kappa = j * param.block_size + 1;
        int end   = min(num_rows, kappa + param.block_size - 1);
        hkz(dummy_kappa_max, param, kappa, end);
      }
      print_tour(i, 0, num_rows);
    }
    catch (RedStatus &e)
    {
      return set_status(e);
    }
  }

  if (flags & BKZ_DUMP_GSO)
  {
    std::ostringstream prefix;
    prefix << "Output ";
    prefix << " (" << std::fixed << std::setw(9) << std::setprecision(3)
           << (cputime() - cputime_start) * 0.001 << "s)";
    dump_gso(param.dump_gso_filename, prefix.str());
  }
  return set_status(final_status);
}

template <class FT> void BKZReduction<FT>::print_tour(const int loop, int min_row, int max_row)
{
  FT r0;
  Float fr0;
  long expo;
  r0  = m.get_r_exp(min_row, min_row, expo);
  fr0 = r0.get_d();
  fr0.mul_2si(fr0, expo);
  cerr << "End of " << algorithm << " loop " << std::setw(4) << loop << ", time = " << std::fixed
       << std::setw(9) << std::setprecision(3) << (cputime() - cputime_start) * 0.001 << "s";
  cerr << ", r_" << min_row << " = " << fr0;
  cerr << ", slope = " << std::setw(9) << std::setprecision(6)
       << m.get_current_slope(min_row, max_row);
  cerr << ", log2(nodes) = " << std::setw(9) << std::setprecision(6) << log2(nodes) << endl;
}

template <class FT> void BKZReduction<FT>::print_after_svp(bool dual, int max_row, int block_size)
{
  FT r0;
  Float fr0;
  long expo;
  r0  = m.get_r_exp(0, 0, expo);
  fr0 = r0.get_d();
  fr0.mul_2si(fr0, expo);

  if (fr0 < best_so_far)
    best_so_far = fr0;

  if (dual)
    num_dsvp ++;
  else
    num_svp ++;

  svp_bs_count[block_size-1]++;
  double tot = (cputime() - cputime_start) * 0.001;

  if (tot > time_so_far) {
    time_so_far = time_so_far + 0.1;
    cerr << "# nsvp " << std::setw( 5 ) << (num_svp+num_dsvp) <<" ("
         << svp_bs_count[input_block_size-1] << ") " <<"t_tot " 
         << std::fixed << std::setw(6) << std::setprecision(2)
         << tot;
    cerr << " t_aux " << std::fixed << std::setw(6) << std::setprecision(2)
         << cputime_others * 0.001;
    cerr << " ( lll " << std::fixed << std::setw(4) << std::setprecision(2)
         << cputime_others_lll * 0.001;
    cerr << " , lll_pre " << std::fixed << std::setw(4) << std::setprecision(2)
         << cputime_others_lll_svppre * 0.001;
    cerr << " , lll_post " << std::fixed << std::setw(4) << std::setprecision(2)
         << cputime_others_lll_svppost * 0.001;
    cerr << " , lll_red " << std::fixed << std::setw(4) << std::setprecision(2)
         << cputime_others_lll_svpred1 * 0.001;
    cerr << " , other " << std::fixed << std::setw(4) << std::setprecision(2)
         << cputime_others_random * 0.001 << " )";
    cerr << " t_svp " << std::fixed << std::setw(6) << std::setprecision(2)
         << cputime_svp * 0.001;
    cerr << " lg " << std::setw(5) << std::setprecision(2)
         << log2(nodes);
    cerr << " slp " << std::setw(7) << std::setprecision(5)
         << m.get_current_slope(0, max_row);
    cerr << " r_" << 0 << " " << fr0 << " r " << best_so_far << endl;
    //cerr << svp_bs_count << endl;
  }
  
  if (tot > 3600 * 30) {
    cerr << "# nsvp_last " << std::setw( 5 ) << (num_svp+num_dsvp) <<" ("
         << svp_bs_count[input_block_size-1] << ") " <<"t_tot "    
         << std::fixed << std::setw(6) << std::setprecision(2)
         << tot;
    cerr << " t_aux " << std::fixed << std::setw(6) << std::setprecision(2)
         << cputime_others * 0.001;
    cerr << " ( " << std::fixed << std::setw(4) << std::setprecision(2)
         << cputime_others_lll * 0.001;
    cerr << " , " << std::fixed << std::setw(4) << std::setprecision(2)
         << cputime_others_random * 0.001 << " )";
    cerr << " t_svp " << std::fixed << std::setw(6) << std::setprecision(2)
         << cputime_svp * 0.001;
    cerr << " lg " << std::setw(5) << std::setprecision(2)
         << log2(nodes);
    cerr << " slp " << std::setw(7) << std::setprecision(5)
         << m.get_current_slope(0, max_row);
    cerr << " r_" << 0 << " " << fr0 << " r " << best_so_far << endl;
    //cerr << svp_bs_count << endl;
    exit(1);
  }
}

template <class FT> void BKZReduction<FT>::print_params(const BKZParam &param, ostream &out)
{
  out << "block size: " << std::setw(3) << param.block_size << ", ";
  out << "flags: 0x" << std::setw(4) << setfill('0') << std::hex << param.flags << ", " << std::dec
      << std::setfill(' ');
  out << "max_loops: " << std::setw(3) << param.max_loops << ", ";
  out << "max_time: " << std::setw(0) << std::fixed << std::setprecision(1) << param.max_time
      << ", ";
  if (param.flags & BKZ_AUTO_ABORT)
  {
    out << "autoAbort: (" << std::setw(0) << std::fixed << std::setprecision(4)
        << param.auto_abort_scale;
    out << ", " << std::setw(2) << param.auto_abort_max_no_dec << "), ";
  }
  else
  {
    out << "autoAbort: (     -,  -), ";
  }
  out << endl;
}

template <class FT> bool BKZReduction<FT>::set_status(int new_status)
{
  status = new_status;
  if (param.flags & BKZ_VERBOSE)
  {
    if (status == RED_SUCCESS)
      cerr << "End of " << algorithm << ": success" << endl;
    else
      cerr << "End of " << algorithm << ": failure: " << RED_STATUS_STR[status] << endl;
  }
  return status == RED_SUCCESS;
}

template <class FT>
void BKZReduction<FT>::dump_gso(const std::string &filename, const std::string &prefix, bool append)
{
  ofstream dump;
  if (append)
    dump.open(filename.c_str(), std::ios_base::app);
  else
    dump.open(filename.c_str());
  dump << std::setw(4) << prefix << ": ";
  FT f, log_f;
  long expo;
  for (int i = 0; i < num_rows; i++)
  {
    m.update_gso_row(i);
    f = m.get_r_exp(i, i, expo);
    log_f.log(f, GMP_RNDU);
    dump << std::setprecision(8) << log_f.get_d() + expo * std::log(2.0) << " ";
  }
  dump << std::endl;
  dump.close();
}

template <class FT> bool BKZAutoAbort<FT>::test_abort(double scale, int maxNoDec)
{
  double new_slope = -m.get_current_slope(start_row, num_rows);
  if (no_dec == -1 || new_slope < scale * old_slope)
    no_dec = 0;
  else
    no_dec++;
  old_slope = min(old_slope, new_slope);
  return no_dec >= maxNoDec;
}

/**
 * call LLLReduction() and then BKZReduction.
 */

template <class FT>
int bkz_reduction_f(IntMatrix &b, const BKZParam &param, int sel_ft, double lll_delta, IntMatrix &u,
                    IntMatrix &u_inv)
{
  int gso_flags = 0;
  if (b.get_rows() == 0 || b.get_cols() == 0)
    return RED_SUCCESS;
  if (sel_ft == FT_DOUBLE || sel_ft == FT_LONG_DOUBLE)
    gso_flags |= GSO_ROW_EXPO;
  MatGSO<Integer, FT> m_gso(b, u, u_inv, gso_flags);
  LLLReduction<Integer, FT> lll_obj(m_gso, lll_delta, LLL_DEF_ETA, LLL_DEFAULT);
  BKZReduction<FT> bkz_obj(m_gso, lll_obj, param);
  bkz_obj.bkz();
  return bkz_obj.status;
}

/**
 * interface called from call_bkz() from main.cpp.
 */
int bkz_reduction(IntMatrix *B, IntMatrix *U, const BKZParam &param, FloatType float_type,
                  int precision)
{
  IntMatrix empty_mat;
  IntMatrix &u     = U ? *U : empty_mat;
  IntMatrix &u_inv = empty_mat;
  FPLLL_CHECK(B, "B == NULL in bkzReduction");

  if (U && (!u.empty()))
  {
    u.gen_identity(B->get_rows());
  }

  double lll_delta = param.delta < 1 ? param.delta : LLL_DEF_DELTA;

  FloatType sel_ft = (float_type != FT_DEFAULT) ? float_type : FT_DOUBLE;
  FPLLL_CHECK(!(sel_ft == FT_MPFR && precision == 0),
              "Missing precision for BKZ with floating point type mpfr");

  /* lllwrapper (no FloatType needed, -m ignored) */
  if (param.flags & BKZ_NO_LLL)
    zeros_last(*B, u, u_inv);
  else
  {
    Wrapper wrapper(*B, u, u_inv, lll_delta, LLL_DEF_ETA, LLL_DEFAULT);
    if (!wrapper.lll())
      return wrapper.status;
  }

  /* bkz (with float_type) */
  int status;
  if (sel_ft == FT_DOUBLE)
  {
    status = bkz_reduction_f<FP_NR<double>>(*B, param, sel_ft, lll_delta, u, u_inv);
  }
#ifdef FPLLL_WITH_LONG_DOUBLE
  else if (sel_ft == FT_LONG_DOUBLE)
  {
    status = bkz_reduction_f<FP_NR<long double>>(*B, param, sel_ft, lll_delta, u, u_inv);
  }
#endif
#ifdef FPLLL_WITH_DPE
  else if (sel_ft == FT_DPE)
  {
    status = bkz_reduction_f<FP_NR<dpe_t>>(*B, param, sel_ft, lll_delta, u, u_inv);
  }
#endif
#ifdef FPLLL_WITH_QD
  else if (sel_ft == FT_DD)
  {
    status = bkz_reduction_f<FP_NR<dd_real>>(*B, param, sel_ft, lll_delta, u, u_inv);
  }
  else if (sel_ft == FT_QD)
  {
    status = bkz_reduction_f<FP_NR<qd_real>>(*B, param, sel_ft, lll_delta, u, u_inv);
  }
#endif
  else if (sel_ft == FT_MPFR)
  {
    int old_prec = FP_NR<mpfr_t>::set_prec(precision);
    status       = bkz_reduction_f<FP_NR<mpfr_t>>(*B, param, sel_ft, lll_delta, u, u_inv);
    FP_NR<mpfr_t>::set_prec(old_prec);
  }
  else
  {
    FPLLL_ABORT("Compiled without support for BKZ reduction with " << FLOAT_TYPE_STR[sel_ft]);
  }
  zeros_first(*B, u, u_inv);
  return status;
}

/**
 * We define BKZ/HKZ for each input type instead of using a template,
 * in order to force the compiler to instantiate the functions.
 */
int bkz_reduction(IntMatrix &b, int block_size, int flags, FloatType float_type, int precision)
{
  vector<Strategy> strategies;
  BKZParam param(block_size, strategies);
  param.flags = flags;
  return bkz_reduction(&b, NULL, param, float_type, precision);
}

int bkz_reduction(IntMatrix &b, IntMatrix &u, int block_size, int flags, FloatType float_type,
                  int precision)
{
  vector<Strategy> strategies;
  BKZParam param(block_size, strategies);
  param.flags = flags;
  return bkz_reduction(&b, &u, param, float_type, precision);
}

int hkz_reduction(IntMatrix &b, int flags, FloatType float_type, int precision)
{
  vector<Strategy> strategies;
  BKZParam param(b.get_rows(), strategies);
  param.block_size = b.get_rows();
  param.delta      = 1;
  if (flags & HKZ_VERBOSE)
    param.flags |= BKZ_VERBOSE;
  return bkz_reduction(&b, NULL, param, float_type, precision);
}

/** enforce instantiation of complete templates **/

template class BKZReduction<FP_NR<double>>;
template class BKZAutoAbort<FP_NR<double>>;

#ifdef FPLLL_WITH_LONG_DOUBLE
template class BKZReduction<FP_NR<long double>>;
template class BKZAutoAbort<FP_NR<long double>>;
#endif

#ifdef FPLLL_WITH_DPE
template class BKZReduction<FP_NR<dpe_t>>;
template class BKZAutoAbort<FP_NR<dpe_t>>;
#endif

#ifdef FPLLL_WITH_QD
template class BKZReduction<FP_NR<dd_real>>;
template class BKZAutoAbort<FP_NR<dd_real>>;

template class BKZReduction<FP_NR<qd_real>>;
template class BKZAutoAbort<FP_NR<qd_real>>;
#endif

template class BKZReduction<FP_NR<mpfr_t>>;
template class BKZAutoAbort<FP_NR<mpfr_t>>;

FPLLL_END_NAMESPACE
