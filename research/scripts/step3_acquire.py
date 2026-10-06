"""Acquire fresh source records in bounded, resumable partitions."""
import argparse
import json
from step3_io import ROOT, OUT, RAW, connect, extract, save_json

FEATURES = json.loads((ROOT / "output/step2_20260910/source_settings.json").read_text())["features"]
IDENTIFIERS = ["id", "permno", "permco", "gvkey", "iid", "excntry", "source_crsp", "eom", "date", "common", "exch_main", "primary_sec", "obs_main", "crsp_shrcd", "crsp_exchcd", "sic", "ff49", "size_grp", "me", "me_company", "shares", "prc", "dolvol", "rvol_252d", "ret", "ret_exc", "ret_exc_lead1m", "ret_lag_dif", "intrinsic_value"]
COLS = list(dict.fromkeys(IDENTIFIERS + FEATURES))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["sample", "monthly", "events", "daily"])
    parser.add_argument("--first-year", type=int, default=1952)
    parser.add_argument("--last-year", type=int, default=2025)
    args = parser.parse_args()
    conn = connect()
    if args.stage == "sample":
        extract(conn, "sample_jkp_nyse_1978_1982", f"SELECT {','.join(COLS)} FROM contrib.global_factor WHERE excntry='USA' AND source_crsp=1 AND crsp_exchcd=1 AND eom BETWEEN %s AND %s ORDER BY eom,id", ["1978-01-31", "1982-12-31"])
        extract(conn, "sample_crsp_monthly_1978_1982", "SELECT permno,date,ret,retx,prc,shrout,vol,cfacpr,cfacshr FROM crsp.msf WHERE date BETWEEN %s AND %s ORDER BY date,permno", ["1978-01-01", "1982-12-31"])
        extract(conn, "crsp_monthly_delistings", "SELECT * FROM crsp.msedelist WHERE dlstdt BETWEEN %s AND %s ORDER BY dlstdt,permno", ["1952-01-01", "2025-12-31"])
        extract(conn, "crsp_mcti", "SELECT caldt,t30ret FROM crsp.mcti WHERE caldt BETWEEN %s AND %s ORDER BY caldt", ["1952-01-01", "2025-12-31"])
        extract(conn, "ff_monthly", "SELECT date,dateff,mktrf,rf FROM ff.factors_monthly WHERE date BETWEEN %s AND %s ORDER BY date", ["1952-01-01", "2025-12-31"])
        extract(conn, "crsp_market_monthly", "SELECT date,vwretd,vwretx FROM crsp.msi WHERE date BETWEEN %s AND %s ORDER BY date", ["1952-01-01", "2025-12-31"])
    elif args.stage == "monthly":
        for year in range(args.first_year, args.last_year + 1):
            start = f"{year}-01-01" if year != 1952 else "1952-12-01"
            end = f"{year}-12-31"
            extract(conn, f"jkp_nyse_{year}", f"SELECT {','.join(COLS)} FROM contrib.global_factor WHERE excntry='USA' AND source_crsp=1 AND crsp_exchcd=1 AND eom BETWEEN %s AND %s ORDER BY eom,id", [start, end])
            # All CRSP securities retain outcomes across exchange changes and disappearance.
            extract(conn, f"crsp_ciz_monthly_{year}", "SELECT permno,permco,mthcaldt,mthprcdt,mthret,mthretx,mthretflg,mthdelflg,mthcompflg,mthcompsubflg,mthprc,mthprcflg,mthcap,mthprevdt,mthprevcap,mthvol,mthcumfacpr,mthcumfacshr,shrout,primaryexch,conditionaltype,tradingstatusflg,securitytype,securitysubtype,sharetype,issuertype FROM crsp.msf_v2 WHERE mthcaldt BETWEEN %s AND %s ORDER BY mthcaldt,permno", [start, end])
    elif args.stage == "events":
        extract(conn, "crsp_ciz_distributions", "SELECT * FROM crsp.stkdistributions WHERE disexdt BETWEEN %s AND %s ORDER BY disexdt,permno,disseqnbr", ["1952-01-01", "2025-12-31"])
        extract(conn, "crsp_ciz_security_names", "SELECT * FROM crsp.stksecurityinfohist WHERE secinfostartdt<=%s AND secinfoenddt>=%s ORDER BY permno,secinfostartdt", ["2025-12-31", "1952-01-01"])
        extract(conn, "crsp_daily_delistings", "SELECT * FROM crsp.dsedelist WHERE dlstdt BETWEEN %s AND %s ORDER BY dlstdt,permno", ["1952-01-01", "2025-12-31"])
        extract(conn, "crsp_security_names", "SELECT permno,namedt,nameendt,shrcd,exchcd,siccd,ncusip,ticker,comnam,permco FROM crsp.msenames WHERE namedt<=%s AND nameendt>=%s ORDER BY permno,namedt", ["2025-12-31", "1952-01-01"])
        extract(conn, "crsp_distribution_events", "SELECT permno,date,distcd,divamt,facpr,facshr,acperm,accomp,dclrdt,rcrddt,paydt FROM crsp.mse WHERE event='DIST' AND date BETWEEN %s AND %s ORDER BY date,permno,distcd", ["1952-01-01", "2025-12-31"])
    elif args.stage == "daily":
        ids = json.loads((OUT / "required_daily_ids.json").read_text())
        for year in range(args.first_year, args.last_year + 1):
            start = f"{year}-01-01" if year != 1952 else "1952-12-01"
            extract(conn, f"crsp_ciz_daily_{year}", "SELECT permno,dlycaldt,dlyret,dlyretx,dlyprc,dlyprcflg,dlyvol,dlycap,dlycumfacshr,dlydelflg,dlyretmissflg,dlyretdurflg FROM crsp.dsf_v2 WHERE permno=ANY(%s) AND dlycaldt BETWEEN %s AND %s ORDER BY dlycaldt,permno", [ids, start, f"{year}-12-31"], chunk_rows=50000)
    conn.close()


if __name__ == "__main__":
    main()
