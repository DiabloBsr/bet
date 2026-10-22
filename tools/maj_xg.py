"""Régénère le bundle xG-forme (scripts/data/xg_form.json).

Understat ne sert ses données qu'à un VRAI NAVIGATEUR (requêtes brutes et IP du
Space bloquées). On NE peut donc PAS fetcher en Python. La récupération se fait
via le navigateur intégré de Claude, avec ce snippet (en-tête X-Requested-With
OBLIGATOIRE), qui rend directement {chemin ESPN: {équipe: xG-forme}} :

    const HDR={'X-Requested-With':'XMLHttpRequest'};
    async function xform(code,yr,n=6){
      const r=await fetch(`https://understat.com/getLeagueData/${code}/${yr}`,{headers:HDR});
      if(!r.ok) return {}; const T=(await r.json()).teams||{}; const o={};
      for(const id in T){const h=(T[id].history||[]).slice(-n); if(!h.length)continue;
        let s=0; for(const m of h){s+=m.xG/((m.xG+m.xGA)||1);}
        o[T[id].title]=+(s/h.length).toFixed(4);}
      return o;}
    const LIG={'soccer/eng.1':'EPL','soccer/esp.1':'La_liga','soccer/ita.1':'Serie_A',
               'soccer/ger.1':'Bundesliga','soccer/fra.1':'Ligue_1'};
    const res={}; for(const p in LIG) res[p]=await xform(LIG[p], <ANNEE_SAISON>);
    return res;

Colle le résultat dans un fichier JSON et lance :
    python tools/maj_xg.py <fichier_xform.json>
puis redéploie. À refaire ~1×/semaine pour garder la xG-forme à jour.
"""
import datetime
import json
import os
import sys

_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "scripts", "data", "xg_form.json")
_SRC = "understat.com/getLeagueData (xG des 6 derniers matchs, ratio xG/(xG+xGA))"


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage : python tools/maj_xg.py <fichier_xform.json "
              "(sortie du snippet navigateur)>")
        sys.exit(1)
    with open(sys.argv[1], encoding="utf-8") as f:
        xform = json.load(f)
    if not isinstance(xform, dict) or not xform:
        print("Fichier invalide : attendu {chemin: {équipe: xG-forme}}.")
        sys.exit(1)
    bundle = {"genere_le": datetime.date.today().isoformat(),
              "source": _SRC, "xform": xform}
    with open(_OUT, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    n = sum(len(v) for v in xform.values())
    print(f"OK : {len(xform)} ligues, {n} équipes -> {_OUT}. Pense à redéployer.")


if __name__ == "__main__":
    main()
