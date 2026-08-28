"""
Extract full column taxonomy and category counts.
"""
import os
import pandas as pd
import json

WORKSPACE = r"c:\Users\Zyren\Documents\orvexa"

def extract_taxonomy():
    raw_path = os.path.join(WORKSPACE, "data", "raw", "esa", "train_data.csv")
    df = pd.read_csv(raw_path, nrows=1000)
    
    # Categorize all 103 columns
    cols = list(df.columns)
    
    id_cols = ['event_id', 'mission_id']
    target_cols = ['risk']
    esa_baseline = ['max_risk_estimate', 'max_risk_scaling']
    kinematics = ['time_to_tca', 'miss_distance', 'relative_speed', 
                  'relative_position_r', 'relative_position_t', 'relative_position_n',
                  'relative_velocity_r', 'relative_velocity_t', 'relative_velocity_n']
    geocentric = ['geocentric_latitude', 'azimuth', 'elevation']
    orbital_elements = ['t_j2k_sma', 't_j2k_ecc', 't_j2k_inc', 'c_j2k_sma', 'c_j2k_ecc', 'c_j2k_inc',
                        't_h_apo', 't_h_per', 'c_h_apo', 'c_h_per']
    object_meta = ['c_object_type', 't_rcs_estimate', 'c_rcs_estimate', 
                   't_cd_area_over_mass', 'c_cd_area_over_mass', 't_cr_area_over_mass', 'c_cr_area_over_mass',
                   't_sedr', 'c_sedr']
    od_quality = ['t_time_lastob_start', 't_time_lastob_end', 't_recommended_od_span', 't_actual_od_span',
                  't_obs_available', 't_obs_used', 't_residuals_accepted', 't_weighted_rms', 't_span',
                  'c_time_lastob_start', 'c_time_lastob_end', 'c_recommended_od_span', 'c_actual_od_span',
                  'c_obs_available', 'c_obs_used', 'c_residuals_accepted', 'c_weighted_rms', 'c_span']
    pos_cov_sigmas = ['t_sigma_r', 't_sigma_t', 't_sigma_n', 'c_sigma_r', 'c_sigma_t', 'c_sigma_n', 'mahalanobis_distance']
    vel_cov_sigmas = ['t_sigma_rdot', 't_sigma_tdot', 't_sigma_ndot', 'c_sigma_rdot', 'c_sigma_tdot', 'c_sigma_ndot']
    cov_dets = ['t_position_covariance_det', 'c_position_covariance_det']
    cov_cross_t = ['t_ct_r', 't_cn_r', 't_cn_t', 't_crdot_r', 't_crdot_t', 't_crdot_n',
                   't_ctdot_r', 't_ctdot_t', 't_ctdot_n', 't_ctdot_rdot', 't_cndot_r', 't_cndot_t',
                   't_cndot_n', 't_cndot_rdot', 't_cndot_tdot']
    cov_cross_c = ['c_ct_r', 'c_cn_r', 'c_cn_t', 'c_crdot_r', 'c_crdot_t', 'c_crdot_n',
                   'c_ctdot_r', 'c_ctdot_t', 'c_ctdot_n', 'c_ctdot_rdot', 'c_cndot_r', 'c_cndot_t',
                   'c_cndot_n', 'c_cndot_rdot', 'c_cndot_tdot']
    space_weather = ['F10', 'F3M', 'SSN', 'AP']
    
    all_cat = (id_cols + target_cols + esa_baseline + kinematics + geocentric + 
               orbital_elements + object_meta + od_quality + pos_cov_sigmas + 
               vel_cov_sigmas + cov_dets + cov_cross_t + cov_cross_c + space_weather)
    
    print(f"Total columns in categorized list: {len(all_cat)} / {len(cols)}")
    missing = set(cols) - set(all_cat)
    print(f"Uncategorized columns: {missing}")

if __name__ == "__main__":
    extract_taxonomy()
