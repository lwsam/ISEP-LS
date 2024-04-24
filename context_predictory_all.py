"""
Uses a long context prediction setting for GPT-3.
"""

import os
import json
import regex
from collections import defaultdict
from typing import List, Tuple, Dict

from langdetect import detect

from tqdm import tqdm
import openai
import re
import unicodedata
from typing import List
from config import API_KEY



# Function to detect the language of the input text
def detect_language(text: str) -> str:
    try:
        detected_language = detect(text)
    except:
        detected_language = "en"  # Default to English if language detection fails
    return detected_language

def clean_predictions(text: str, given_word: str) -> List[str]:
    """
    Post-processing of files, by trying different strategies to coerce it into actual singular predictions.
    :param text: Unfiltered text predicted by a language model
    :param given_word: The word that is supposed to be replaced. Sometimes appears in `text`.
    :return: List of individual predictions
    """

    # Catch sample 248
    if text.startswith(given_word):
        text = text[len(given_word):]

    # Clear additional clutter that might have been encountered
    text = text.strip("\n :;.?!")

    try:
        # Presence of newlines within the prediction indicates prediction as list
        if "\n" in text.strip("\n "):
            cleaned_predictions = text.strip("\n ").split("\n")

        # Other common format contained comma-separated list without anything else
        elif "," in text.strip("\n "):
            cleaned_predictions = [pred.strip(" ") for pred in text.strip("\n ").split(",")]

        # Sometimes in-line enumerations also occur, this is a quick check to more or less guarantee
        # at least 6 enumerated predictions
        elif re.search(r'\b[1-9]\.', text):
            cleaned_predictions = re.split(r'[0-9]{1,2}\.', text.strip("\n "))

        
        # Handle special case with hyphen-separated predictions
        elif '-' in text:
            cleaned_predictions = [pred.strip() for pred in text.split("-")]

        else:
            raise ValueError(f"Unrecognized list format in prediction '{text}'")

        # Edge case where there is inconsistent newlines
        if len(cleaned_predictions) < 2:
            raise ValueError(f"Inconsistent newline pattern found in prediction '{text}'")

    except ValueError as e:
        print(f"Skipping prediction: {e}")
        return []

    # Remove numerals
    cleaned_predictions = [remove_numerals(pred) for pred in cleaned_predictions]
    # Make sure everything is lower-cased and stripped
    cleaned_predictions = [pred.lower().strip(" \n") for pred in cleaned_predictions]
    # Remove "to" in the beginning
    cleaned_predictions = [remove_to(pred) for pred in cleaned_predictions]
    # Remove predictions that match the given word
    cleaned_predictions = remove_identity_predictions(cleaned_predictions, given_word)
    # Remove empty predictions that may have slipped through:
    cleaned_predictions = remove_empty_predictions(cleaned_predictions)
    # Remove multi-word predictions (with 3 or more steps)
    cleaned_predictions = remove_multiwords(cleaned_predictions)
    # Remove punctuation
    cleaned_predictions = remove_punctuation(cleaned_predictions)

    return cleaned_predictions



def remove_punctuation(predictions: List[str]) -> List[str]:
    return [prediction.strip(".,;?!") for prediction in predictions]

def remove_multiwords(predictions: List[str], max_segments: int = 2) -> List[str]:
    return [prediction for prediction in predictions if len(prediction.split(" ")) <= max_segments]

def remove_empty_predictions(predictions: List[str]) -> List[str]:
    return [pred for pred in predictions if pred.strip("\n ")]

def remove_identity_predictions(predictions: List[str], given_word: str) -> List[str]:
    return [pred for pred in predictions if pred != given_word]

def remove_numerals(text: str) -> str:
    """
    Will remove any leading numerals (optionally with a dot).
    :param text: Input text, potentially containing a leading numeral
    :return: cleaned text
    """
    return regex.sub(r"[0-9]{1,2}\.? ?", "", text)

def remove_to(text: str) -> str:
    """
    Removes the leading "to"-infinitive from a prediction, which is sometimes caused when the context word
    is preceeded with a "to" in the text.
    :param text: Prediction text
    :return: Text where a leading "to " would be removed from the string.
    """
    return regex.sub(r"^to ", "", text)

def deduplicate_predictions(predictions: List[Tuple]) -> Dict:
    """
    Slightly less efficient deduplication method that preserves "ranking order" by appearance.
    :param predictions: List of predictions
    :return: Filtered list of predictions that no longer contains duplicates.
    """
    merged = defaultdict(float)
    for prediction, score in predictions:
        merged[prediction] += score

    return merged

def get_highest_predictions(predictions: Dict, number_predictions: int) -> List[str]:
    return [prediction for prediction, _ in sorted(predictions.items(), key=lambda item: item[1], reverse=True)][:number_predictions]

def assign_prediction_scores(predictions: List[str], start_weight: float = 5.0, decrease: float = 0.5) -> List[Tuple]:
    """
    The result of   predictions - len(predictions) * decrease   should equal 0.
    :param predictions:
    :param start_weight:
    :param decrease:
    :return:
    """
    weighted_predictions = []
    for idx, prediction in enumerate(predictions):
        weighted_predictions.append((prediction, start_weight - idx * decrease))

    return weighted_predictions

def get_prompts_and_temperatures(context: str, word: str, language: str) -> List[Tuple[str, str, float]]:
    zero_shot = f"Context: {context}\n" \
                f"Question: Given the above context, list ten alternative words for \"{word}\" that are easier to understand.\n" \
                f"Answer:"

    no_context_zero_shot = f"Give me ten simplified synonyms for the following word: {word}"

    no_context_single_shot = f"Question: Find ten easier words for \"compulsory\".\n" \
                             f"Answer:\n" \
                             f"1. mandatory\n2. required\n3. essential\n4. forced\n5. important\n" \
                             f"6. necessary\n7. obligatory\n8. unavoidable\n9. binding\n10. prescribed\n\n" \
                             f"Question: Find ten easier words for \"{word}\".\n" \
                             f"Answer:"

    single_shot_prompt = f"Context: A local witness said a separate group of attackers disguised in burqas — the head-to-toe robes worn by conservative Afghan women — then tried to storm the compound.\n" \
                         f"Question: Given the above context, list ten alternative words for \"disguised\" that are easier to understand.\n" \
                         f"Answer:\n" \
                         f"1. concealed\n2. dressed\n3. hidden\n4. camouflaged\n5. changed\n" \
                         f"6. covered\n7. masked\n8. unrecognizable\n9. converted\n10. impersonated\n\n"\
                         f"Context: {context}\n" \
                         f"Question: Given the above context, list ten alternative words for \"{word}\" that are easier to understand.\n" \
                         f"Answer:"

    few_shot_prompt = f"Context: That prompted the military to deploy its largest warship, the BRP Gregorio del Pilar, which was recently acquired from the United States.\n" \
                      f"Question: Given the above context, list ten alternative words for \"deploy\" that are easier to understand.\n" \
                      f"Answer:\n" \
                      f"1. send\n2. post\n3. use\n4. position\n5. send out\n" \
                      f"6. employ\n7. extend\n8. launch\n9. let loose\n10. organize\n\n" \
                      f"Context: The daily death toll in Syria has declined as the number of observers has risen, but few experts expect the U.N. plan to succeed in its entirety.\n" \
                      f"Question: Given the above context, list ten alternative words for \"observers\" that are easier to understand.\n" \
                      f"Answer:\n" \
                      f"1. watchers\n2. spectators\n3. audience\n4. viewers\n5. witnesses\n" \
                      f"6. patrons\n7. followers\n8. detectives\n9. reporters\n10. onlookers\n\n" \
                      f"Context: {context}\n" \
                      f"Question: Given the above context, list ten alternative words for \"{word}\" that are easier to understand.\n" \
                      f"Answer:"

    # Mix between different methods
    prompts = [("conservative zero-shot with context", zero_shot, 0.3),
               ("creative zero-shot with context", zero_shot, 0.8),
               ("zero-shot without context", no_context_zero_shot, 0.7),
               ("single-shot without context", no_context_single_shot, 0.6),
               ("single-shot with context", single_shot_prompt, 0.5),
               ("few-shot with context", few_shot_prompt, 0.5)]

    # Language-specific prompts and temperatures
    if language == "ca":  # Catalan
        zero_shot_ca = f"Context: {context}\n" \
                    f"Question: Basant-vos en el context anterior, llisteu deu paraules alternatives per a \"{word}\" que siguin més fàcils de comprendre.\n" \
                    f"Resposta:"

        no_context_zero_shot_ca = f"Donam deu sinònims simplificats per a la següent paraula: {word}"

        no_context_single_shot_ca = f"Pregunta: Trobeu deu paraules més fàcils per a \"compulsory\".\n" \
                                    f"Resposta:\n" \
                                    f"1. obligatori\n2. necessari\n3. imprescindible\n4. forçat\n5. important\n" \
                                    f"6. inexcusable\n7. prescindible\n8. imposat\n9. imperatiu\n10. inexorable\n\n" \
                                    f"Pregunta: Trobeu deu paraules més fàcils per a \"{word}\".\n" \
                                    f"Resposta:"

        single_shot_prompt_ca = f"Context: Un testimoni local va dir que un grup separat d'atacadors disfressats de burkas — els vestits de cap a peus que porten les dones afganes conservadores — van intentar després d'irrompre en el recinte.\n" \
                                f"Pregunta: Basant-vos en el context anterior, lliste-ho deu paraules alternatives per a \"disfressats\" que siguin més fàcils de comprendre.\n" \
                                f"Resposta:\n" \
                                f"1. camuflat\n2. encobert\n3. amagat\n4. disfressat\n5. canviat\n" \
                                f"6. tapat\n7. disimulat\n8. desconegut\n9. convertit\n10. personificat\n\n" \
                                f"Context: {context}\n" \
                                f"Pregunta: Basant-vos en el context anterior, lliste-ho deu paraules alternatives per a \"{word}\" que siguin més fàcils de comprendre.\n" \
                                f"Resposta:"

        few_shot_prompt_ca = f"Context: Això va provocar que el militar desplegués el seu vaixell de guerra més gran, el BRP Gregorio del Pilar, que va ser adquirit recentment pels Estats Units.\n" \
                            f"Pregunta: Basant-vos en el context anterior, llisteu deu paraules alternatives per a \"desplegar\" que siguin més fàcils de comprendre.\n" \
                            f"Resposta:\n" \
                            f"1. enviar\n2. posar\n3. utilitzar\n4. col·locar\n5. expedir\n" \
                            f"6. emprar\n7. estendre\n8. llançar\n9. alliberar\n10. organitzar\n\n" \
                            f"Context: La taxa diària de morts a Síria ha disminuït a mesura que el nombre d'observadors ha augmentat, però pocs experts esperen que el pla de l'ONU tingui èxit en la seva totalitat.\n" \
                            f"Pregunta: Basant-vos en el context anterior, llisteu deu paraules alternatives per a \"observadors\" que siguin més fàcils de comprendre.\n" \
                            f"Resposta:\n" \
                            f"1. espectadors\n2. vigiles\n3. observadors\n4. testimonis\n5. espectacle\n" \
                            f"6. públic\n7. convidats\n8. detectives\n9. informadors\n10. guardonadors\n\n" \
                            f"Context: {context}\n" \
                            f"Pregunta: Basant-vos en el context anterior, lliste-ho deu paraules alternatives per a \"{word}\" que siguin més fàcils de comprendre.\n" \
                            f"Resposta:"

        # Adjust prompts and temperatures for Catalan
        prompts_ca = [("conservative zero-shot with context", zero_shot_ca, 0.3),
                    ("creative zero-shot with context", zero_shot_ca, 0.8),
                    ("zero-shot without context", no_context_zero_shot_ca, 0.7),
                    ("single-shot without context", no_context_single_shot_ca, 0.6),
                    ("single-shot with context", single_shot_prompt_ca, 0.5),
                    ("few-shot with context", few_shot_prompt_ca, 0.5)]

        return prompts_ca

    elif language == "en":  # English
        return prompts
    elif language == "fil":  # Filipino
        zero_shot_fil = f"Konteksto: {context}\n" \
                        f"Tanong: Batay sa nakalista na konteksto, maglista ng sampung alternatibong mga salita para sa \"{word}\" na mas madali maintindihan.\n" \
                        f"Sagot:"

        no_context_zero_shot_fil = f"Ibigay mo ang sampung pinasimple na mga sinonimo para sa sumusunod na salita: {word}"

        no_context_single_shot_fil = f"Tanong: Hanapin mo ang sampung mas madaling mga salita para sa \"compulsory\".\n" \
                                    f"Sagot:\n" \
                                    f"1. obligatorio\n2. kailangan\n3. mahalaga\n4. sapilitan\n5. importante\n" \
                                    f"6. kinakailangan\n7. kailangang-kailangan\n8. hindi maiiwasan\n9. kagustuhan\n10. utos\n\n" \
                                    f"Tanong: Hanapin mo ang sampung mas madaling mga salita para sa \"{word}\".\n" \
                                    f"Sagot:"

        single_shot_prompt_fil = f"Konteksto: Ayon sa isang lokal na saksi, isang hiwalay na grupo ng mga manlalaban na nagdisguise sa mga burka — ang head-to-toe na mga kasuotang isinusuot ng mga konserbatibong kababaihang Afghan — ay sumubok na saksakin ang compound.\n" \
                                f"Tanong: Batay sa nakalista na konteksto, maglista ng sampung alternatibong mga salita para sa \"nagdisguise\" na mas madali maintindihan.\n" \
                                f"Sagot:\n" \
                                f"1. nagtago\n2. nagbihis\n3. nagtago\n4. nagbalatkayo\n5. nagbago\n" \
                                f"6. nakatago\n7. nagmaskara\n8. hindi nakikilalang\n9. nag-iba\n10. nagtampok\n\n" \
                                f"Konteksto: {context}\n" \
                                f"Tanong: Batay sa nakalista na konteksto, maglista ng sampung alternatibong mga salita para sa \"{word}\" na mas madali maintindihan.\n" \
                                f"Sagot:"

        few_shot_prompt_fil = f"Konteksto: Ito ang nagtulak sa militar na ideploy ang kanilang pinakamalaking barko ng digmaan, ang BRP Gregorio del Pilar, na kamakailan ay inakwir mula sa Estados Unidos.\n" \
                            f"Tanong: Batay sa nakalista na konteksto, maglista ng sampung alternatibong mga salita para sa \"ideploy\" na mas madali maintindihan.\n" \
                            f"Sagot:\n" \
                            f"1. ipadala\n2. mag-post\n3. gamitin\n4. maglagay\n5. magpadala\n" \
                            f"6. mag-employ\n7. mag-extend\n8. maglunsad\n9. magpakawala\n10. mag-organisa\n\n" \
                            f"Konteksto: Ang araw-araw na bilang ng mga namamatay sa Syria ay bumaba habang dumarami ang bilang ng mga tagamasid, ngunit ilang mga eksperto ang umaasa na ang plano ng UN ay magtatagumpay sa kabuuan nito.\n" \
                            f"Tanong: Batay sa nakalista na konteksto, maglista ng sampung alternatibong mga salita para sa \"tagamasid\" na mas madali maintindihan.\n" \
                            f"Sagot:\n" \
                            f"1. mga tagapanood\n2. mga manonood\n3. audience\n4. mga tagasilip\n5. mga saksi\n" \
                            f"6. mga tagabantay\n7. mga tagasubaybay\n8. mga tagasaliksik\n9. mga tagapagbalita\n10. mga tagapanood\n\n" \
                            f"Konteksto: {context}\n" \
                            f"Tanong: Batay sa nakalista na konteksto, maglista ng sampung alternatibong mga salita para sa \"{word}\" na mas madali maintindihan.\n" \
                            f"Sagot:"

        # Adjust prompts and temperatures for Filipino
        prompts_fil = [("conservative zero-shot with context", zero_shot_fil, 0.3),
                    ("creative zero-shot with context", zero_shot_fil, 0.8),
                    ("zero-shot without context", no_context_zero_shot_fil, 0.7),
                    ("single-shot without context", no_context_single_shot_fil, 0.6),
                    ("single-shot with context", single_shot_prompt_fil, 0.5),
                    ("few-shot with context", few_shot_prompt_fil, 0.5)]

        return prompts_fil

    elif language == "fr":  # French
        zero_shot_fr = f"Contexte: {context}\n" \
                    f"Question: En vous basant sur le contexte ci-dessus, listez dix mots alternatifs pour \"{word}\" qui sont plus faciles à comprendre.\n" \
                    f"Réponse:"

        no_context_zero_shot_fr = f"Donnez-moi dix synonymes simplifiés pour le mot suivant : {word}"

        no_context_single_shot_fr = f"Question: Trouvez dix mots plus simples pour \"obligatoire\".\n" \
                                    f"Réponse:\n" \
                                    f"1. nécessaire\n2. requis\n3. essentiel\n4. imposé\n5. important\n" \
                                    f"6. indispensable\n7. obligatoire\n8. inévitable\n9. contraignant\n10. prescrit\n\n" \
                                    f"Question: Trouvez dix mots plus simples pour \"{word}\".\n" \
                                    f"Réponse:"

        single_shot_prompt_fr = f"Contexte : Un témoin local a déclaré qu'un groupe séparé d'assaillants déguisés en burqas - les robes de la tête aux pieds portées par les femmes afghanes conservatrices - a ensuite tenté de prendre d'assaut le bâtiment.\n" \
                                f"Question : En vous basant sur le contexte ci-dessus, listez dix mots alternatifs pour \"déguisés\" qui sont plus faciles à comprendre.\n" \
                                f"Réponse:\n" \
                                f"1. cachés\n2. vêtus\n3. dissimulés\n4. camouflés\n5. changés\n" \
                                f"6. couverts\n7. masqués\n8. méconnaissables\n9. transformés\n10. usurpés\n\n" \
                                f"Contexte : {context}\n" \
                                f"Question : En vous basant sur le contexte ci-dessus, listez dix mots alternatifs pour \"{word}\" qui sont plus faciles à comprendre.\n" \
                                f"Réponse:"

        few_shot_prompt_fr = f"Contexte : Cela a incité l'armée à déployer son plus grand navire de guerre, le BRP Gregorio del Pilar, qui a récemment été acquis auprès des États-Unis.\n" \
                            f"Question : En vous basant sur le contexte ci-dessus, listez dix mots alternatifs pour \"déployer\" qui sont plus faciles à comprendre.\n" \
                            f"Réponse:\n" \
                            f"1. envoyer\n2. poster\n3. utiliser\n4. positionner\n5. envoyer\n" \
                            f"6. employer\n7. étendre\n8. lancer\n9. libérer\n10. organiser\n\n" \
                            f"Contexte : Le nombre quotidien de décès en Syrie a diminué à mesure que le nombre d'observateurs augmentait, mais peu d'experts s'attendent à ce que le plan de l'ONU réussisse dans son ensemble.\n" \
                            f"Question : En vous basant sur le contexte ci-dessus, listez dix mots alternatifs pour \"observateurs\" qui sont plus faciles à comprendre.\n" \
                            f"Réponse:\n" \
                            f"1. spectateurs\n2. témoins\n3. auditeurs\n4. observateurs\n5. spectatrices\n" \
                            f"6. surveillants\n7. assistants\n8. témoins oculaires\n9. rapporteurs\n10. espions\n\n" \
                            f"Contexte : {context}\n" \
                            f"Question : En vous basant sur le contexte ci-dessus, listez dix mots alternatifs pour \"{word}\" qui sont plus faciles à comprendre.\n" \
                            f"Réponse:"

        # Adjust prompts and temperatures for French
        prompts_fr = [("conservative zero-shot with context", zero_shot_fr, 0.3),
                    ("creative zero-shot with context", zero_shot_fr, 0.8),
                    ("zero-shot without context", no_context_zero_shot_fr, 0.7),
                    ("single-shot without context", no_context_single_shot_fr, 0.6),
                    ("single-shot with context", single_shot_prompt_fr, 0.5),
                    ("few-shot with context", few_shot_prompt_fr, 0.5)]

        return prompts_fr

    elif language == "de":  # German
        # Adjust prompts and temperatures for German
        # Define German prompts and temperatures
        zero_shot_de = f"Kontext: {context}\n" \
                    f"Frage: Basierend auf dem obigen Kontext, listen Sie zehn alternative Wörter für \"{word}\" auf, die einfacher zu verstehen sind.\n" \
                    f"Antwort:"

        no_context_zero_shot_de = f"Gib mir zehn vereinfachte Synonyme für das folgende Wort: {word}"

        no_context_single_shot_de = f"Frage: Finde zehn einfachere Wörter für \"obligatorisch\".\n" \
                                f"Antwort:\n" \
                                f"1. verpflichtend\n2. erforderlich\n3. wesentlich\n4. erzwungen\n5. wichtig\n" \
                                f"6. notwendig\n7. obligatorisch\n8. unvermeidbar\n9. bindend\n10. vorgeschrieben\n\n" \
                                f"Frage: Finde zehn einfachere Wörter für \"{word}\".\n" \
                                f"Antwort:"

        single_shot_prompt_de = f"Kontext: Ein örtlicher Zeuge sagte, eine separate Gruppe von Angreifern, die in Burkas verkleidet waren - den von konservativen afghanischen Frauen getragenen Ganzkörperschleiern - habe dann versucht, das Gelände zu stürmen.\n" \
                            f"Frage: Basierend auf dem obigen Kontext, listen Sie zehn alternative Wörter für \"verkleidet\" auf, die einfacher zu verstehen sind.\n" \
                            f"Antwort:\n" \
                            f"1. versteckt\n2. gekleidet\n3. versteckt\n4. getarnt\n5. verändert\n" \
                            f"6. bedeckt\n7. maskiert\n8. unerkennbar\n9. umgewandelt\n10. dargestellt\n\n" \
                            f"Kontext: {context}\n" \
                            f"Frage: Basierend auf dem obigen Kontext, listen Sie zehn alternative Wörter für \"{word}\" auf, die einfacher zu verstehen sind.\n" \
                            f"Antwort:"

        few_shot_prompt_de = f"Kontext: Das veranlasste die Armee, ihr größtes Kriegsschiff, die BRP Gregorio del Pilar, einzusetzen, das kürzlich aus den Vereinigten Staaten erworben wurde.\n" \
                        f"Frage: Basierend auf dem obigen Kontext, listen Sie zehn alternative Wörter für \"einsetzen\" auf, die einfacher zu verstehen sind.\n" \
                        f"Antwort:\n" \
                        f"1. senden\n2. posten\n3. verwenden\n4. positionieren\n5. aussenden\n" \
                        f"6. anstellen\n7. ausdehnen\n8. starten\n9. loslassen\n10. organisieren\n\n" \
                        f"Kontext: Die tägliche Todesrate in Syrien ist gesunken, da die Anzahl der Beobachter gestiegen ist, aber nur wenige Experten erwarten, dass der U.N.-Plan in seiner Gesamtheit erfolgreich sein wird.\n" \
                        f"Frage: Basierend auf dem obigen Kontext, listen Sie zehn alternative Wörter für \"Beobachter\" auf, die einfacher zu verstehen sind.\n" \
                        f"Antwort:\n" \
                        f"1. Zuschauer\n2. Publikum\n3. Zuschauer\n4. Zuschauer\n5. Zeugen\n" \
                        f"6. Kunden\n7. Anhänger\n8. Detektive\n9. Reporter\n10. Zuschauer\n\n" \
                        f"Kontext: {context}\n" \
                        f"Frage: Basierend auf dem obigen Kontext, listen Sie zehn alternative Wörter für \"{word}\" auf, die einfacher zu verstehen sind.\n" \
                        f"Antwort:"

        # Mix between different methods
        prompts_de = [("conservative zero-shot with context", zero_shot_de, 0.3),
                ("creative zero-shot with context", zero_shot_de, 0.8),
                ("zero-shot without context", no_context_zero_shot_de, 0.7),
                ("single-shot without context", no_context_single_shot_de, 0.6),
                ("single-shot with context", single_shot_prompt_de, 0.5),
                ("few-shot with context", few_shot_prompt_de, 0.5)]

        return prompts_de

    elif language == "it":  # Italian
        # Adjust prompts and temperatures for Italian
        # Define Italian prompts and temperatures
        zero_shot_it = f"Contesto: {context}\n" \
                    f"Domanda: Dato il contesto sopra, elenca dieci parole alternative per \"{word}\" che sono più facili da capire.\n" \
                    f"Risposta:"

        no_context_zero_shot_it = f"Dammi dieci sinonimi semplificati per la parola seguente: {word}"

        no_context_single_shot_it = f"Domanda: Trova dieci parole più facili per \"obbligatorio\".\n" \
                                f"Risposta:\n" \
                                f"1. obbligatorio\n2. richiesto\n3. essenziale\n4. forzato\n5. importante\n" \
                                f"6. necessario\n7. obbligatorio\n8. inevitabile\n9. vincolante\n10. prescritto\n\n" \
                                f"Domanda: Trova dieci parole più facili per \"{word}\".\n" \
                                f"Risposta:"

        single_shot_prompt_it = f"Contesto: Un testimone locale ha detto che un gruppo separato di aggressori travestiti con burqa - i veli integrali indossati dalle donne afghane conservatrici - ha poi cercato di assaltare il complesso.\n" \
                            f"Domanda: Dato il contesto sopra, elenca dieci parole alternative per \"travestito\" che sono più facili da capire.\n" \
                            f"Risposta:\n" \
                            f"1. nascosto\n2. vestito\n3. celato\n4. mimetizzato\n5. cambiato\n" \
                            f"6. coperto\n7. mascherato\n8. irriconoscibile\n9. convertito\n10. impersonato\n\n" \
                            f"Contesto: {context}\n" \
                            f"Domanda: Dato il contesto sopra, elenca dieci parole alternative per \"{word}\" che sono più facili da capire.\n" \
                            f"Risposta:"

        few_shot_prompt_it = f"Contesto: Ciò ha indotto l'esercito a schierare la sua più grande nave da guerra, la BRP Gregorio del Pilar, che è stata recentemente acquisita dagli Stati Uniti.\n" \
                        f"Domanda: Dato il contesto sopra, elenca dieci parole alternative per \"schierare\" che sono più facili da capire.\n" \
                        f"Risposta:\n" \
                        f"1. inviare\n2. pubblicare\n3. utilizzare\n4. posizionare\n5. inviare\n" \
                        f"6. impiegare\n7. estendere\n8. lanciare\n9. rilasciare\n10. organizzare\n\n" \
                        f"Contesto: Il numero giornaliero di morti in Siria è diminuito man mano che il numero di osservatori è aumentato, ma pochi esperti si aspettano che il piano dell'ONU riesca interamente.\n" \
                        f"Domanda: Dato il contesto sopra, elenca dieci parole alternative per \"osservatori\" che sono più facili da capire.\n" \
                        f"Risposta:\n" \
                        f"1. spettatori\n2. pubblico\n3. osservatori\n4. spettatori\n5. testimoni\n" \
                        f"6. clienti\n7. seguaci\n8. investigatori\n9. giornalisti\n10. curiosi\n\n" \
                        f"Contesto: {context}\n" \
                        f"Domanda: Dato il contesto sopra, elenca dieci parole alternative per \"{word}\" che sono più facili da capire.\n" \
                        f"Risposta:"

        # Mix between different methods
        prompts_it = [("conservative zero-shot with context", zero_shot_it, 0.3),
                ("creative zero-shot with context", zero_shot_it, 0.8),
                ("zero-shot without context", no_context_zero_shot_it, 0.7),
                ("single-shot without context", no_context_single_shot_it, 0.6),
                ("single-shot with context", single_shot_prompt_it, 0.5),
                ("few-shot with context", few_shot_prompt_it, 0.5)]

        return prompts_it

    elif language == "ja":  # Japanese
        # Adjust prompts and temperatures for Japanese
        # Define Japanese prompts and temperatures
        zero_shot_ja = f"文脈: {context}\n" \
                    f"質問: 上記の文脈を考慮して、「{word}」の理解しやすい代替語を10個リストアップしてください。\n" \
                    f"回答:"

        no_context_zero_shot_ja = f"以下の単語に対する10の簡略化された類義語を教えてください：{word}"

        no_context_single_shot_ja = f"質問: 「強制的」という言葉のより簡単な言葉を10個見つけてください。\n" \
                                f"回答:\n" \
                                f"1. 強制\n2. 必要\n3. 必須\n4. 必要不可欠\n5. 必要\n" \
                                f"6. 絶対\n7. 必修\n8. 必要な\n9. 強制された\n10. 義務的\n\n" \
                                f"質問: 「{word}」のより簡単な言葉を10個見つけてください。\n" \
                                f"回答:"

        single_shot_prompt_ja = f"文脈: 現地の目撃者は、保守的なアフガン女性が着用する全身ローブであるブルカで変装した別の攻撃者グループが、その後、施設を襲おうとしたと述べた。\n" \
                            f"質問: 上記の文脈を考慮して、「変装した」のより理解しやすい代替語を10個リストアップしてください。\n" \
                            f"回答:\n" \
                            f"1. 隠れた\n2. 着飾った\n3. 隠れた\n4. カモフラージュした\n5. 変化した\n" \
                            f"6. 覆われた\n7. マスクされた\n8. 識別できない\n9. 変換された\n10. 偽装された\n\n" \
                            f"文脈: {context}\n" \
                            f"質問: 上記の文脈を考慮して、「{word}」の理解しやすい代替語を10個リストアップしてください。\n" \
                            f"回答:"

        few_shot_prompt_ja = f"文脈: それにより、軍は米国から最近取得した最大の戦艦、BRP Gregorio del Pilarを展開しました。\n" \
                        f"質問: 上記の文脈を考慮して、「展開する」の理解しやすい代替語を10個リストアップしてください。\n" \
                        f"回答:\n" \
                        f"1. 送る\n2. 投稿する\n3. 使用する\n4. 位置する\n5. 出荷する\n" \
                        f"6. 運用する\n7. 拡張する\n8. 打ち上げる\n9. 解放する\n10. 組織する\n\n" \
                        f"文脈: シリアの毎日の死者数は、観察者の数が増えるにつれて減少していますが、ほとんどの専門家は、国連の計画が完全に成功するとは予想していません。\n" \
                        f"質問: 上記の文脈を考慮して、「観察者」の理解しやすい代替語を10個リストアップしてください。\n" \
                        f"回答:\n" \
                        f"1. 観客\n2. 見物人\n3. 見学者\n4. 鑑賞者\n5."

        # Mix between different methods
        prompts_ja = [("conservative zero-shot with context", zero_shot_ja, 0.3),
                ("creative zero-shot with context", zero_shot_ja, 0.8),
                ("zero-shot without context", no_context_zero_shot_ja, 0.7),
                ("single-shot without context", no_context_single_shot_ja, 0.6),
                ("single-shot with context", single_shot_prompt_ja, 0.5),
                ("few-shot with context", few_shot_prompt_ja, 0.5)]

        return prompts_ja

    elif language == "pt":  # Portuguese
        # Adjust prompts and temperatures for Portuguese
        zero_shot_pt = f"Contexto: {context}\n" \
                    f"Pergunta: Com base no contexto acima, liste dez palavras alternativas para \"{word}\" que sejam mais fáceis de entender.\n" \
                    f"Resposta:"

        no_context_zero_shot_pt = f"Dê-me dez sinônimos simplificados para a seguinte palavra: {word}"

        no_context_single_shot_pt = f"Pergunta: Encontre dez palavras mais fáceis para \"obrigatório\".\n" \
                                f"Resposta:\n" \
                                f"1. obrigatório\n2. necessário\n3. essencial\n4. forçado\n5. importante\n" \
                                f"6. indispensável\n7. impreterível\n8. preciso\n9. vital\n10. compulsório\n\n" \
                                f"Pergunta: Encontre dez palavras mais fáceis para \"{word}\".\n" \
                                f"Resposta:"

        single_shot_prompt_pt = f"Contexto: Uma testemunha local disse que um grupo separado de atacantes disfarçados de burcas - as roupas de corpo inteiro usadas por mulheres afegãs conservadoras - tentou então invadir o complexo.\n" \
                            f"Pergunta: Com base no contexto acima, liste dez palavras alternativas para \"disfarçados\" que sejam mais fáceis de entender.\n" \
                            f"Resposta:\n" \
                            f"1. disfarçados\n2. mascarados\n3. camuflados\n4. ocultos\n5. encobertos\n" \
                            f"6. escondidos\n7. travestidos\n8. encapotados\n9. dissimulados\n10. enfeitados\n\n" \
                            f"Contexto: {context}\n" \
                            f"Pergunta: Com base no contexto acima, liste dez palavras alternativas para \"{word}\" que sejam mais fáceis de entender.\n" \
                            f"Resposta:"

        few_shot_prompt_pt = f"Contexto: Isso levou a marinha a desdobrar seu maior navio de guerra, o BRP Gregorio del Pilar, que foi recentemente adquirido dos Estados Unidos.\n" \
                        f"Pergunta: Com base no contexto acima, liste dez palavras alternativas para \"desdobrar\" que sejam mais fáceis de entender.\n" \
                        f"Resposta:\n" \
                        f"1. desdobrar\n2. posicionar\n3. estender\n4. desenrolar\n5. implantar\n" \
                        f"6. movimentar\n7. empregar\n8. implementar\n9. enviar\n10. direcionar\n\n" \
                        f"Contexto: O número diário de mortes na Síria diminuiu à medida que o número de observadores aumentou, mas poucos especialistas esperam que o plano da ONU tenha sucesso em sua totalidade.\n" \
                        f"Pergunta: Com base no contexto acima, liste dez palavras alternativas para \"observadores\" que sejam mais fáceis de entender.\n" \
                        f"Resposta:\n" \
                        f"1. observadores\n2. espectadores\n3. testemunhas\n4. olheiros\n5. guardiões\n" \
                        f"6. vigilantes\n7. espiões\n8. informantes\n9. agentes\n10. monitoradores\n\n" \
                        f"Contexto: {context}\n" \
                        f"Pergunta: Com base no contexto acima, liste dez palavras alternativas para \"{word}\" que sejam mais fáceis de entender.\n" \
                        f"Resposta:"

        # Mix between different methods
        prompts_pt = [("conservative zero-shot with context", zero_shot_pt, 0.3),
                ("creative zero-shot with context", zero_shot_pt, 0.8),
                ("zero-shot without context", no_context_zero_shot_pt, 0.7),
                ("single-shot without context", no_context_single_shot_pt, 0.6),
                ("single-shot with context", single_shot_prompt_pt, 0.5),
                ("few-shot with context", few_shot_prompt_pt, 0.5)]

        return prompts_pt

    elif language == "si":  # Sinhala
        # Adjust prompts and temperatures for Sinhala
        zero_shot_si = f"සංකේත: {context}\n" \
                    f"විස්තරය: ඉහලින් දැක්කා ඇති අතර, \"{word}\" සඳහා ප්‍රියතම ව්‍යාකරණ වචන 10 යි ලබාදෙන්න.\n" \
                    f"පිළිතුරු:"

        no_context_zero_shot_si = f"ප‍්‍රශ්නය: {word} සඳහා අයිතියේ හදුනා ගැනීමට පියවර පැහැදිලි සංඛ්‍යාව ලබාදෙන්න."

        no_context_single_shot_si = f"ප‍්‍රශ්නය: \"අනුපූර්ණ\" සඳහා අයිතියේ හදුනා ගැනීමට අනුවාදයන් අස්ථර්භ කරන්න.\n" \
                                f"පිළිතුරු:\n" \
                                f"1. අනුපූර්ණ\n2. විචල්කලා\n3. විචල්ක\n4. ආරම්භ\n5. විශ්වාස\n" \
                                f"6. අනුපූර්ණයි\n7. අත්සන\n8. අත්පොත්\n9. හැරුනුම්\n10. අනුප්‍රේල්ශික\n\n" \
                                f"ප‍්‍රශ්නය: \"{word}\" සඳහා අයිතියේ හදුනා ගැනීමට අනුවාදයන් අස්ථර්භ කරන්න.\n" \
                                f"පිළිතුරු:"

        single_shot_prompt_si = f"ප‍්‍රශ්නය: ඉහලින් දැක්කා ඇති අතර, \"{word}\" සඳහා ප්‍රියතම ව්‍යාකරණ වචන 10 යි ලබාදෙන්න.\n" \
                        f"පිළිතුරු:"

        few_shot_prompt_si = f"සංකේත: ඒ පිටුවට යෙදෙන්නේ විශාලතම මෘදුකාංගයක්, BRP ග්‍රෙගොරියෝ ඩෙල් පිලර්, මෙමගින් එක්ව ගත් එකතුවක්.\n" \
                    f"ප‍්‍රශ්නය: ඉහලින් දැක්කා ඇති අතර, \"deploy\" සඳහා ප්‍රියතම ව්‍යාකරණ වචන 10 යි ලබාදෙන්න.\n" \
                    f"පිළිතුරු:\n" \
                    f"1. යවනා\n2. තැන්පතුම්\n3. භාවිතය\n4. ස්ථිර\n5. යවන්නේ\n" \
                    f"6. ඉහලින් ක්‍රියා කරන්න\n7. දිස්වෙනවා\n8. පෙන්වන්න\n9. නික්මෙන්න\n10. සාරාංශගත කරන්න\n\n" \
                    f"සංකේත: සියතලිත පොලිසියට සපත්තු වර්ෂයේ අවස්වැඩි වේදිකාවෙන් මරණ ගණන අඩු වෙයි, " \
                    f"ඇය අනුපාතිවරයා නොවේ, තවත් විදේශීයවන නිර්මාන සංවර්ධනය සාරාංශ කිරීම පිලිබඳ සිදුවෙමි.\n" \
                    f"ප‍්‍රශ්නය: ඉහලින් දැක්කා ඇති අතර, \"{word}\" සඳහා ප්‍රියතම ව්‍යාකරණ වචන 10 යි ලබාදෙන්න.\n" \
                    f"පිළිතුරු:"

        # Mix between different methods
        prompts_si = [("conservative zero-shot with context", zero_shot_si, 0.3),
                        ("creative zero-shot with context", zero_shot_si, 0.8),
                        ("zero-shot without context", no_context_zero_shot_si, 0.7),
                        ("single-shot without context", no_context_single_shot_si, 0.6),
                        ("single-shot with context", single_shot_prompt_si, 0.5),
                        ("few-shot with context", few_shot_prompt_si, 0.5)]

        return prompts_si


    elif language == "es":  # Spanish
        # Adjust prompts and temperatures for Spanish
        zero_shot_es = f"Contexto: {context}\n" \
                    f"Pregunta: Dado el contexto anterior, enumera diez palabras alternativas para \"{word}\" que sean más fáciles de entender.\n" \
                    f"Respuesta:"

        no_context_zero_shot_es = f"Dame diez sinónimos simplificados para la siguiente palabra: {word}"

        no_context_single_shot_es = f"Pregunta: Encuentra diez palabras más fáciles para \"obligatorio\".\n" \
                                f"Respuesta:\n" \
                                f"1. necesario\n2. imprescindible\n3. obligado\n4. forzoso\n5. imperativo\n" \
                                f"6. fundamental\n7. esencial\n8. requerido\n9. preceptivo\n10. ineludible\n\n" \
                                f"Pregunta: Encuentra diez palabras más fáciles para \"{word}\".\n" \
                                f"Respuesta:"

        single_shot_prompt_es = f"Contexto: Un testigo local dijo que un grupo separado de atacantes, disfrazados con burkas — los trajes de pies a cabeza usados por mujeres afganas conservadoras — luego intentaron asaltar el complejo.\n" \
                            f"Pregunta: Dado el contexto anterior, enumera diez palabras alternativas para \"disfrazados\" que sean más fáciles de entender.\n" \
                            f"Respuesta:\n" \
                            f"1. camuflados\n2. ocultos\n3. encubiertos\n4. vestidos\n5. enmascarados\n" \
                            f"6. escondidos\n7. disimulados\n8. maquillados\n9. transformados\n10. mimetizados\n\n" \
                            f"Contexto: {context}\n" \
                            f"Pregunta: Dado el contexto anterior, enumera diez palabras alternativas para \"{word}\" que sean más fáciles de entender.\n" \
                            f"Respuesta:"

        few_shot_prompt_es = f"Contexto: Eso llevó a que la armada desplegara su mayor nave de guerra, el BRP Gregorio del Pilar, que fue adquirida recientemente de los Estados Unidos.\n" \
                        f"Pregunta: Dado el contexto anterior, enumera diez palabras alternativas para \"desplegar\" que sean más fáciles de entender.\n" \
                        f"Respuesta:\n" \
                        f"1. enviar\n2. colocar\n3. usar\n4. distribuir\n5. movilizar\n" \
                        f"6. posicionar\n7. desatar\n8. organizar\n9. emplear\n10. ejecutar\n\n" \
                        f"Contexto: La cantidad diaria de muertes en Siria ha disminuido a medida que ha aumentado el número de observadores, pero pocos expertos esperan que el plan de la ONU tenga éxito en su totalidad.\n" \
                        f"Pregunta: Dado el contexto anterior, enumera diez palabras alternativas para \"observadores\" que sean más fáciles de entender.\n" \
                        f"Respuesta:\n" \
                        f"1. testigos\n2. espectadores\n3. miradores\n4. vigilantes\n5. observantes\n" \
                        f"6. espectadores\n7. fisgones\n8. curiosos\n9. informantes\n10. asistentes\n\n" \
                        f"Contexto: {context}\n" \
                        f"Pregunta: Dado el contexto anterior, enumera diez palabras alternativas para \"{word}\" que sean más fáciles de entender.\n" \
                        f"Respuesta:"

        # Mix between different methods
        prompts_es = [("conservative zero-shot with context", zero_shot_es, 0.3),
                            ("creative zero-shot with context", zero_shot_es, 0.8),
                            ("zero-shot without context", no_context_zero_shot_es, 0.7),
                            ("single-shot without context", no_context_single_shot_es, 0.6),
                            ("single-shot with context", single_shot_prompt_es, 0.5),
                            ("few-shot with context", few_shot_prompt_es, 0.5)]

        return prompts_es

    else:
        print(f"No prompts available for language '{language}'. Defaulting to English prompts.")
        return prompts  # Default to English prompts

if __name__ == '__main__':
    debug = False
    max_number_predictions = 10
    continue_from = 0

    if debug:
        with open("D:/Telechargements/TSAR-2022-Shared-Task-main/TSAR-2022-Shared-Task-main/datasets/test/test2/multilex_test_de_ls_unlabelled.tsv", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        with open("D:/Telechargements/TSAR-2022-Shared-Task-main/TSAR-2022-Shared-Task-main/datasets/test/test2/multilex_test_de_ls_unlabelled.tsv", encoding="utf-8") as f:
            lines = f.readlines()

    openai.api_key = API_KEY

    baseline_predictions = []
    ensemble_predictions = []

    if debug:
        lines = lines[:2]

    for idx, line in enumerate(tqdm(lines)):
        # Skip already processed samples
        if idx < continue_from:
            continue

        aggregated_predictions = []

        # Extract context and complex word
        line_values = line.strip("\n ").split("\t")
        if len(line_values) != 2:
            continue  # Ignore incorrectly formatted lines
        context, word = line_values

        # Get language-specific prompts and temperatures
        prompts_and_temps = get_prompts_and_temperatures(context, word, "en")  # Defaulting to English prompts

        # Get language-specific prompts and temperatures based on the detected language
        detected_language = detect_language(context)
        if detected_language in ["ca", "en", "fil", "fr", "de", "it", "ja", "pt", "si", "es"]:
            prompts_and_temps = get_prompts_and_temperatures(context, word, detected_language)
        else:
            print(f"Unsupported language detected: {detected_language}. Defaulting to English prompts.")

        for prompt_name, prompt, temperature in tqdm(prompts_and_temps):
            # Have not experimented too much with other parameters, but these generally worked well.
            response = openai.Completion.create(
                model="gpt-3.5-turbo-instruct-0914",
                prompt=prompt,
                stream=False,
                temperature=temperature,
                max_tokens=256,
                top_p=1,
                best_of=1,
                frequency_penalty=0.5,
                presence_penalty=0.3
            )
            predictions = response["choices"][0]["text"]

            predictions = clean_predictions(predictions, word)
            weighted_predictions = assign_prediction_scores(predictions)
            aggregated_predictions.extend(weighted_predictions)

            # Store the "conservative zero-shot with context" predictions as a baseline run.
            if prompt_name == "conservative zero-shot with context":
                baseline_predictions.append(weighted_predictions)
                with open("tsar2022_test_en_UniHD_1.tsv", "a") as f:
                    prediction_string = "\t".join(predictions[:max_number_predictions])
                    f.write(f"{context}\t{word}\t{prediction_string}\n")

        aggregated_predictions = deduplicate_predictions(aggregated_predictions)
        highest_scoring_predictions = get_highest_predictions(aggregated_predictions, max_number_predictions)
        with open("tsar2022_test_en_UniHD_3.tsv", "a") as f:
            prediction_string = "\t".join(highest_scoring_predictions[:max_number_predictions])
            f.write(f"{context}\t{word}\t{prediction_string}\n")

       

        

        # Get the best predictions and other relevant data
        best_predictions = get_highest_predictions(aggregated_predictions, max_number_predictions)

        # Write predictions to TSV file
        output_file_path = "D:/Telechargements/TSAR-2022-Shared-Task-main/TSAR-2022-Shared-Task-main/output_predictions.tsv"
        try:
            with open(output_file_path, "a", encoding="utf-8") as output_file:
                prediction_string = "\t".join(best_predictions)
                
                # Check each character in the string for compatibility
                sanitized_string = ""
                for char in f"{context}\t{word}\t{prediction_string}\n":
                    try:
                        # Try to encode the character
                        char.encode('utf-8')
                        # If successful, add it to the sanitized string
                        sanitized_string += char
                    except UnicodeEncodeError:
                        # If the character causes an encoding error, replace it with a blank space
                        sanitized_string += ' '
                        
                # Write the sanitized string to the file
                output_file.write(sanitized_string)
        except Exception as e:
            # Handle any other exceptions
            print(f"Skipping line {idx + 1} due to an error: {e}")
            continue

        ensemble_predictions.append(aggregated_predictions)




        if debug:
            print(f"Complex word: {word}")
            print(f"{aggregated_predictions}")




    print("Baseline Predictions:", baseline_predictions)
    print("Ensemble Predictions:", ensemble_predictions)

    # FIXME: This currently overwrites previously generated scores!!!
    with open("baseline_scores.json", "w") as f:
        json.dump(baseline_predictions, f, ensure_ascii=False, indent=2)
    with open("ensemble_scores.json", "w") as f:
        json.dump(ensemble_predictions, f, ensure_ascii=False, indent=2)
