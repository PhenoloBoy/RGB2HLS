import xarray as xr
import numpy as np
import pandas as pd


def _rgb2hsv(rgb):

    min = rgb.min(dim='bands')
    Max = rgb.max(dim='bands')
    diff = (Max - min)
    zero = ~ np.equal(min, Max)*1

    # masks
    r = np.logical_and(rgb[0] > rgb[2], rgb[0] > rgb[1])
    g = np.logical_and(rgb[1] > rgb[0], rgb[1] > rgb[2])
    b = np.logical_and(rgb[2] > rgb[0], rgb[2] > rgb[1])

    # converted values

    rh = (60 * ((rgb[1] - rgb[2]).where(r) / diff) + 360) % 360
    gh = (60 * ((rgb[2] - rgb[0]).where(g) / diff) + 120) % 360
    bh = (60 * ((rgb[0] - rgb[1]).where(b) / diff) + 240) % 360

    # HSV
    H = xr.concat([rh, gh, bh], 'H').max(dim='H') * zero; H.name = 'H'

    V = Max*100; V.name = 'V'

    S = ((diff/V)*100) * zero; S.name = 'S'

    return xr.Dataset({'H': H, 'S': S, 'V': V})


def _ambiguity_mask(vgt, NDVI):
    H = vgt['H']
    mask = np.logical_and(np.logical_and(H < (NDVI * 255), H > 14.5 + (NDVI * 75)),
                           np.logical_and(H < 8 + (NDVI * 210), H > 60 - (NDVI * 250)))
    return mask


def _NDVI(RED, NIR):
    return (NIR - RED) / (NIR + RED)


def _slope(vgt):
    H = vgt['H']
    median = H.median(axis=0).values
    H_diff = np.diff(H, axis=0, prepend=np.expand_dims(median, 0))
    # time_diff = np.diff(date_da, axis=1)
    slope = H_diff/10
    return slope


def _scatter(NDVI, H):

    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(NDVI, H)
    plt.show()


def main():
    ds = xr.open_mfdataset('c:/data/HLS/samples/Eritrea/*.nc', combine='nested', concat_dim='time')

    B04 = ds.B04
    B03 = ds.B03
    B02 = ds.B02
    B08 = ds.B8A
    B11 = ds.B11

    NDVI = _NDVI(B04, B08)

    rgb = xr.concat([B04, B03, B02], pd.Index(['r', 'g', 'b'], name='bands'))
    rgb.name = 'RGB'

    veg = xr.concat([B11, B08, B04], pd.Index(['r', 'g', 'b'], name='bands'))
    veg.name = 'VGT'

    visible = _rgb2hsv(rgb)

    veg_idx = _rgb2hsv(veg)

    vegetated = veg_idx.H.where(veg_idx.H > 82-NDVI*255)

    ambiguity = _ambiguity_mask(veg_idx, NDVI)
    slope = _slope(veg_idx)
    slope_mean = np.nanmean(slope, axis=0)
    slope_std = np.nanstd(slope, axis=0)

    semi_vegetated = veg_idx.H.where(ambiguity).where(slope > slope_mean + 3 * slope_std)

    index = np.logical_or(vegetated.notnull(), semi_vegetated.notnull())

    index.to_netcdf('c:/data/HLS/result.nc')

    print('GVxL index created')


if __name__ == '__main__':
    main()
