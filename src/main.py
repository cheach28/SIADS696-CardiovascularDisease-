from data.data_Import import transform_data_to_df


if __name__ == "__main__":
    data = transform_data_to_df()

    print(data.head())