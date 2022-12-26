regex_patterns = {
    "VIOLENCIA_DE_GENERO": {
        "fisica": r"[Ff][ií]sic[ao]",
        "psicologica": r"[Pp]sico(?:l[oó]gica)?",
        "economica": r"[Ee]con[oó]mica",
        "social": r"[Ss]ocial",
        "ambiental": r"[Aa]mbiental",
        "simbolica": r"[Ss]imb[oó]lica",
        "sexual": r"[Ss]exual",
        "politica": r"[Pp]ol[ií]tica",
    },
    "MODALIDAD_DE_LA_VIOLENCIA": {
        "domestica": r"[Dd]om[ée]stic[ao]|[Dd]omicili[ao]r?|[Ff]amiliar?",
        "en_espacio_publico": r"[Pp]úblic[oa]",
        "institucional": r"[Ii]nstitucional",
        "laboral": r"[Ll]aboral|[Tt]rabajo",
        "libertad_reproductiva": r"[Ll]ibertad reproductiva|[Rr]eproducci[oó]n",
        "mediatica": r"[Mm]edi[aá](?!nte)(?:tic[ao])?s?",
        "obstetrica": r"[Oo]bt[eé]tric(?:[ao]|ia)",
        "publica_politica": r"[Pp][uú]blic[ao]|[Pp]ol[ií]tic[ao]",
    },
    "RELACION_Y_TIPO_ENTRE_ACUSADO/A_Y_DENUNCIANTE": {
        # "ninguna": r"",
        "familiar": r"(?:familiar?|hij[ao]s?|[mp]adres?|herman[ao]s?)",
        "pareja": r"(?<!ex)(?<!ex )(?:pareja|matrimonio|marid[ao]|espos[ao]|conyuge|amantes?)(?! de la madre)",
        "ex_pareja": r"(?:(?:ex|eran|solian? .+) ?(?:pareja|conyuge|companero|marid[ao]|espos[ao])|separa(?:d[ao]s?|cion|ron|rse))",
        "jefe_empleada": r"(?:jef[ae]|superior|jerarqui(?:a|co)|laboral|trabajo)",
        "familiar_de_ex_pareja": r"(?:ex ?(suegr[ao]|yerno|nuera|cunad[ao])|familiar .+ex ?pareja)",
        "pareja_de_la_madre": r"pareja de (?:la|su) madre",
        "vecino": r"vecin[ao]s?",
        "profesor": r"profesor[ae]?s?|docentes?",
        "propietario_del_inmueble_de_residencia": r"propietari[ao]s?",
        "compañero": r"(?<!ex)(?<!ex )companer[ao]s?",
        "amigo": r"amig[ao]s?",
        "familiar_de_pareja": r"(?<!ex)(?<!ex )(?:familiar .+(?<!ex) ?pareja|suegr[ao]|yerno|nuera|cunad[ao])",
    },
    "TIPO_DE_RESOLUCION": {
        "interlocutoria": r"interlocut[oa]ria",
        "definitiva": r"definitiv[ao]",
    },
    "NIVEL_INSTRUCCION": {
        # "sin_instruccion": r"",
        "sin_escolarizar": r"(?:no (?:posee|termino)|sin).+estudios?",
        "primario_incompleto": r"(?:(?:no (?:termino|cuenta con|concluyo).+(?:estudios?|educacion) ?)primari[ao]s?|primari[ao]s? incomplet[ao]s?)",
        "primario_completo": r"(?:primari[ao]s? (?!in)complet[ao]s?|(?<!continuar con la regularidad de sus )(?<!continua con sus )(?<!cursa sus )(?<!cursando sus )(?<!sin )estudios primarios(?! incomplet[ao]s?)(?! en curso)|(?:(?:termino|cuenta con|concluyo).+estudios? ?)primari[ao]s?)",
        "primario_en_curso": r"(?:primari[ao]s? en curso|cursa.+primari[ao]s?|continua.+primari[ao]s?)",
        "secundario_incompleto": r"(?:(?:no (?:termino|cuenta con|concluyo).+(?:estudios?|educacion) ?)secundari[ao]s?|secundari[ao]s? incomplet[ao]s?)",
        "secundario_completo": r"(?:secundari[ao]s? (?!in)complet[ao]s?|(?<!continuar con la regularidad de sus )(?<!continua con sus )(?<!cursa sus )(?<!cursando sus )(?<!sin )estudios secundarios(?! incomplet[ao]s?)(?! en curso)|(?:(?:termino|cuenta con|concluyo).+estudios? ?)secundari[ao]s?)",
        "secundario_en_curso": r"(?:secundari[ao]s? en curso|cursa.+secundari[ao]s?|continua.+secundari[ao]s?)",
        "terciario_incompleto": r"(?:(?:no (?:termino|cuenta con|concluyo).+(?:estudios?|educacion) ?)terciari[ao]s?|terciari[ao]s? incomplet[ao]s?)",
        "terciario_completo": r"(?:terciari[ao]s? (?!in)complet[ao]s?|(?<!continuar con la regularidad de sus )(?<!continua con sus )(?<!cursa sus )(?<!cursando sus )(?<!sin )estudios terciarios(?! incomplet[ao]s?)(?! en curso)|(?:(?:termino|cuenta con|concluyo).+estudios? ?)terciari[ao]s?)",
        "terciario_en_curso": r"(?:terciari[ao]s? en curso|cursa.+terciari[ao]s?|continua.+terciari[ao]s?)",
        "universitario_incompleto": r"(?:(?:no (?:termino|cuenta con|concluyo).+(?:estudios?|educacion) ?)universitari[ao]s?|universitari[ao]s? incomplet[ao]s?)",
        "universitario_completo": r"(?:universitari[ao]s? (?!in)complet[ao]s?|(?<!continuar con la regularidad de sus )(?<!continua con sus )(?<!cursa sus )(?<!cursando sus )(?<!sin )estudios universitarios(?! incomplet[ao]s?)(?! en curso)|(?:(?:termino|cuenta con|concluyo).+estudios? ?)universitari[ao]s?|post?grado)",
        "universitario_en_curso": r"(?:universitari[ao]s? en curso|cursa.+universitari[ao]s?|continua.+universitari[ao]s?)",
    },
    "GENERO": {
        "varon_cis": r"varon(?:es)? cis",
        "mujer_cis": r"mujer(?:es)? cis",
        "varon_trans": r"varon(?:es)? trans",
        "mujer_trans": r"mujer(?:es)? trans",
        "travesti": r"travesti",
        "no_binaria": r"no ?binari[aoex]",
    },
    "PERSONA_ACUSADA_NO_DETERMINADA": {
        # could be improve using company names ner
        "persona_juridica": r"\W(S\.?R\.?L\.?|S\.?A\.?|sociedad anonima|consorcio|laboratorio|asociacion civil)\W",
        "personal_policial": r"policia",
        # all user categories should be looked somehow explicitely
        "usuario_de_facebook": r"(facebook)",
        "usuario_de_cuenta_de_google": r"(gmail)",
        "usuario_de_instagram": r"(instagram)",
        "usuario_de_twitter": r"(twitter)",
        "usuario_de_outlook": r"(hotmail)",
        "usuario_de_skype": r"(skype)",
        "usuario_microsoft": r"(microsoft|hotmail|one drive)",
        "usuario_de_whatsapp": r"(whatsapp|telefonia|celular)",
        "usuario_de_youtube": r"(you\s?tube)",
        "usuario_de_mercado_libre": r"(mercado\s?libre)",
        # general url pattern (https://stackoverflow.com/a/3809435)
        "pagina_web": r"(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)",
    },
    "NOMBRE": {
        "firma": r"juez",
    },
}
