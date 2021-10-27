library(magrittr)

df <- cars %>%
  dplyr::mutate(ln_hpwt = log(hpwt),
                ln_space = log(space),
                ln_mpg = log(mpg),
                ln_mpd = log(mpd),
                ln_price = log(price),
                trend = market + 70,
                cons = 1) %>%
  dplyr::group_by(model_year) %>%
  dplyr::mutate(s_0 = log(1-sum(share))) %>%
  dplyr::ungroup() %>%
  dplyr::mutate(s_i = log(share)) %>%
  dplyr::mutate(dif = s_i - s_0, 
                dif_2 = log(share) - log(share_out),
                ln_price = log(price)) %>%
  dplyr::arrange(market, firmid)
