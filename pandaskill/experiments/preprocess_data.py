from os.path import join

from pandaskill.experiments.data.drop_games import drop_unwanted_games
from pandaskill.experiments.data.player_region import (
    attribute_player_in_game_to_region,
    attribute_player_region_change,
    manually_correct_team_region,
)
from pandaskill.experiments.data.preprocess_data import (
    clean_up_largest_killing_spree,
    clean_up_largest_multi_kill,
    compute_features,
    drop_neutral_objective_events_with_none_killer_id,
    load_raw_data,
)
from pandaskill.experiments.general.utils import ARTIFACTS_DIR, save_yaml

data_dir = join(ARTIFACTS_DIR, "data")


def preprocess_raw_data() -> None:
    raw_data_dir = join(data_dir, "raw")
    stat_df, event_df = load_raw_data(raw_data_dir)
    stat_df, event_df, dropped_games_summary = drop_unwanted_games(stat_df, event_df)

    stat_df = clean_up_largest_killing_spree(stat_df)
    stat_df = clean_up_largest_multi_kill(stat_df)

    initial_columns = stat_df.columns

    stat_df = attribute_player_in_game_to_region(stat_df)
    stat_df = manually_correct_team_region(stat_df)
    stat_df = attribute_player_region_change(stat_df)

    event_df, dropped_event_summary = drop_neutral_objective_events_with_none_killer_id(
        event_df
    )

    drop_log_dir = join(data_dir, "preprocessing", "logs")
    save_yaml(dropped_games_summary, drop_log_dir, "dropped_games.yaml")
    save_yaml(dropped_event_summary, drop_log_dir, "dropped_events.yaml")

    stat_df = compute_features(stat_df, event_df)
    feature_columns = stat_df.columns.difference(initial_columns)
    stat_df.loc[:, feature_columns].to_csv(
        join(data_dir, "preprocessing", "game_features.csv")
    )


if __name__ == "__main__":
    preprocess_raw_data()
