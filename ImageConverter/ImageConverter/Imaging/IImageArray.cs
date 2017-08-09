namespace ImageConverter.Imaging
{
	public interface IImageArray
	{
		/// <summary> Количество каналов изображения. </summary>
		int Channels { get; }

		/// <summary> Массив пиксельных матриц каналов изображения. </summary>
		byte[][][] ChannelsArr { get; }

		/// <summary> Индексатор. </summary>
		/// <param name="channel"> Номер канала, начинается с 0. </param>
		/// <returns> Возвращает матрицу пикселей, соответствущую номеру канала. </returns>
		byte[][] this[int channel] { get; set; }
	}
}
