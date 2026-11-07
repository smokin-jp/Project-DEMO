"""CLIP zero-shot prompt sets per strike type. label -> list of prompt phrasings (embeddings are averaged).

Each strike type ("punch", "kick", "knee") has: contact, location, glove (striking surface), genuine, and for
kick/knee also `check` (is this really a strike, or just a step / lifted leg?). Location labels are shared:
head, torso, below_belt (groin), back, arms_or_gloves (blocked), legs (thigh/shin), knee_joint, none.
"""

# ----------------------------------------------------------------------------------------------- punch
PUNCH = {
    "contact": {
        "contact": [
            "a photo of a boxer's glove hitting an opponent's face",
            "a photo of a boxing glove landing on an opponent's body",
            "a photo of a punch connecting with the other boxer",
        ],
        "no_contact": [
            "a photo of a boxer throwing a punch that misses, with a gap between the glove and the opponent",
            "a photo of a boxer punching at empty air in front of the opponent",
            "a photo of a boxing glove that does not touch the opponent",
        ],
    },
    "location": {
        "head": ["a photo of a boxing glove hitting the front of an opponent's face", "a photo of a punch to the side of the head"],
        "torso": ["a photo of a boxing glove hitting an opponent's chest or ribs", "a photo of a body shot to the stomach above the belt"],
        "below_belt": ["a photo of a boxer punching an opponent below the belt line in the groin", "a photo of a low blow to the hips"],
        "back": ["a photo of a boxer hitting an opponent on the back of the head", "a photo of a boxer punching an opponent in the back or kidney"],
        "arms_or_gloves": ["a photo of a boxing glove hitting the opponent's raised gloves and guard", "a photo of a punch absorbed by the opponent's arms"],
        "legs": ["a photo of a punch hitting the opponent's thigh or leg"],
        "knee_joint": ["a photo of a punch hitting the opponent's knee"],
        "none": ["a photo of two boxers where no punch is touching the opponent", "a photo of a boxer whose glove is in the air away from the opponent"],
    },
    "glove": {
        "knuckle_part": ["a close-up photo of a closed boxing glove punching with the front knuckle part", "a close-up photo of a clenched boxing glove striking with its hitting surface"],
        "open_glove": ["a close-up photo of an open boxing glove slapping with the palm or inside of the glove", "a close-up photo of a boxing glove hitting with its open palm side"],
        "forearm_or_elbow": ["a close-up photo of a boxer hitting with the elbow", "a close-up photo of a boxer striking with the forearm or the wrist"],
    },
    "genuine": {
        "clean": ["a photo of a clean punch landing flush on an opponent", "a photo of a boxer landing a solid punch on the head or body"],
        "block": ["a photo of a punch blocked by the opponent's gloves", "a photo of a boxer covering up and absorbing a punch with the guard"],
        "miss": ["a photo of a punch that misses the opponent", "a photo of a boxer slipping a punch so the glove passes by"],
        "push": ["a photo of a boxer pushing or shoving the opponent with straight arms", "a photo of a boxer shoving the other boxer away"],
        "glancing": ["a photo of a glancing blow barely brushing the opponent", "a photo of a glove grazing past the side of the opponent's head"],
    },
}

# ------------------------------------------------------------------------------------------------ kick
KICK = {
    "contact": {
        "contact": [
            "a photo of a kick landing on an opponent",
            "a photo of a fighter's shin or foot hitting the other fighter",
            "a photo of a roundhouse kick connecting with the opponent",
        ],
        "no_contact": [
            "a photo of a kick that misses the opponent",
            "a photo of a fighter kicking at empty air with a gap before the target",
            "a photo of a kick passing in front of the opponent without touching",
        ],
    },
    "location": {
        "head": ["a photo of a high kick to the opponent's head", "a photo of a kick hitting the side of the head"],
        "torso": ["a photo of a kick to the opponent's body, ribs or stomach", "a photo of a body kick above the belt"],
        "below_belt": ["a photo of a kick to the groin", "a photo of a low kick to the hips below the belt"],
        "back": ["a photo of a kick to the back of the head", "a photo of a kick to the opponent's back"],
        "arms_or_gloves": ["a photo of a kick blocked by the opponent's arms and gloves", "a photo of a kick absorbed by the opponent's guard"],
        "legs": ["a photo of a low kick to the opponent's thigh", "a photo of a leg kick hitting the opponent's leg"],
        "knee_joint": ["a photo of a thrusting kick to the opponent's knee joint", "a photo of a straight kick to the knee"],
        "none": ["a photo of two fighters where no kick touches the opponent", "a photo of a raised leg in the air away from the opponent"],
    },
    "glove": {  # striking surface; foot_or_shin is legal
        "foot_or_shin": ["a close-up photo of a kick landing with the shin", "a close-up photo of a kick landing with the instep or top of the foot"],
        "other": ["a close-up photo of a stomp or a heel jab onto the opponent", "a close-up photo of a fighter pushing with the sole of the foot"],
    },
    "genuine": {
        "clean": ["a photo of a clean kick landing solidly", "a photo of a powerful kick connecting flush on the opponent"],
        "block": ["a photo of a kick blocked by the opponent's raised leg, arm or elbow", "a photo of a kick checked by the opponent's shin"],
        "miss": ["a photo of a kick that misses the opponent", "a photo of a fighter leaning back so a kick passes in front of them"],
        "push": ["a photo of a push kick shoving the opponent away", "a photo of a fighter pushing the opponent with the sole of the foot"],
        "glancing": ["a photo of a kick that barely grazes the opponent", "a photo of a shin brushing past the side of the opponent"],
    },
    "check": {
        "real": ["a photo of a fighter throwing a kick", "a photo of a fighter with one leg raised, striking out"],
        "not_real": ["a photo of a fighter walking or stepping", "a photo of a fighter standing in a stance with both feet on the ground", "a photo of a fighter shuffling their feet"],
    },
}

# ------------------------------------------------------------------------------------------------ knee
KNEE = {
    "contact": {
        "contact": ["a photo of a knee strike landing on an opponent", "a photo of a fighter driving a knee into the opponent's body"],
        "no_contact": ["a photo of a knee strike that misses the opponent", "a photo of a fighter lifting a knee with a gap before the opponent"],
    },
    "location": {
        "head": ["a photo of a knee strike to the opponent's head"],
        "torso": ["a photo of a knee to the opponent's ribs or stomach", "a photo of a knee strike to the body above the belt"],
        "below_belt": ["a photo of a knee strike to the groin"],
        "back": ["a photo of a knee strike to the opponent's back"],
        "arms_or_gloves": ["a photo of a knee strike blocked by the opponent's arms and elbows"],
        "legs": ["a photo of a knee strike to the opponent's thigh"],
        "knee_joint": ["a photo of a knee hitting the opponent's knee"],
        "none": ["a photo of two fighters where no knee touches the opponent"],
    },
    "glove": {  # striking surface; knee is legal
        "knee": ["a close-up photo of a knee strike hitting with the knee"],
        "other": ["a close-up photo of a fighter hitting with the shin or foot instead of the knee"],
    },
    "genuine": {
        "clean": ["a photo of a clean knee strike landing solidly", "a photo of a knee connecting flush on the opponent"],
        "block": ["a photo of a knee strike blocked by the opponent's arms or elbows"],
        "miss": ["a photo of a knee strike that misses the opponent"],
        "push": ["a photo of a fighter pushing the opponent away with the thigh or knee"],
        "glancing": ["a photo of a knee that only grazes the opponent"],
    },
    "check": {
        "real": ["a photo of a fighter driving a knee up into an opponent", "a photo of a fighter lifting a knee to strike"],
        "not_real": ["a photo of a fighter walking or stepping", "a photo of a fighter standing in a stance with both feet on the ground", "a photo of a fighter lifting a leg to step over something"],
    },
}

PROMPT_SETS = {"punch": PUNCH, "kick": KICK, "knee": KNEE}
