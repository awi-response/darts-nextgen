from pathlib import Path

import ee, geedim as gd

# TODO add earth engine login here

def downloadS2data( s2_id:str, download_folder, bands = ["B2","B3","B4","B8"], download_scl=True, ee_img_predl_callback = None ):
    """downloads a whole Sentinel 2 product identified by s2_id as well as its SCL band as a seperate file. The resulting
    TIF files will be in the native CRS of the product which is the UTM zone appropiate for the location.

    The ee_img_predl_callback will be called after instantiation of the ee.Images of the SR and SCL data. Additional operations
    can be applied here, most notably a clip() to the images if necessary. Must return an ee.Image instance.
    See the implementation of downloadS2Clipped() for details.

    Args:
        s2_id (str): the Sentinel 2 ID like '20230807T202851_20230807T203151_T10WEE'
        download_folder (str): the folder where to download to. Both tif files (SR and SCL) will be placed here
        bands (list, optional): the bands to include, must be the band names according to the GEE catalog. Defaults to ["B2","B3","B4","B8"].
        ee_img_predl_callback (callable, optional): a callback applied to the ee.Image instances of the bands. Defaults to None.

    Returns:
        _type_: _description_
    """

    download_folder = Path(download_folder)

    download_filepath = download_folder / f"{s2_id}_SR.tif"
    sr_ee_img = ee.Image("COPERNICUS/S2_SR_HARMONIZED/"+s2_id).select(bands)
    if callable(ee_img_predl_callback):
        sr_ee_img = ee_img_predl_callback(sr_ee_img)
    gd_sr_img = gd.download.BaseImage(sr_ee_img)


    if download_scl:
        scl_download_filepath = download_folder / f"{s2_id}_SCL.tif"
        scl_ee_img = ee.Image("COPERNICUS/S2_SR_HARMONIZED/"+s2_id).select(["SCL"])
        if callable(ee_img_predl_callback):
            scl_ee_img = ee_img_predl_callback(scl_ee_img)
        gd_scl_img = gd.download.BaseImage(scl_ee_img)

    download_folder.mkdir( parents=True,exist_ok=True )
    gd_sr_img.download( download_filepath, dtype='uint16', overwrite=True)
    if download_scl:
        gd_scl_img.download( scl_download_filepath, dtype='uint16', overwrite=True)
        return download_filepath, scl_download_filepath
    else:
        return download_filepath