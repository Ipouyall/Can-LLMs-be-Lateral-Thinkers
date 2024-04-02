import matplotlib.pyplot as plt

table_CoT_gpt35 = {
    "Direct": {
        "None": 72.5,
        "Compressed": 72.5,
        "Detailed": 75.0,
    },
    "Internal-Simple": {
        "None": 70.0,
        "Compressed": 70.0,
        "Detailed": 72.5,
    },
    "Internal-Specified": {
        "None": 57.5,
        "Compressed": 60.0,
        "Detailed": 62.5,
    },
    "External": {
        "None": 67.5,
        "Compressed": 65.0,
        "Detailed": 62.5,
    }
}

table_rag_gpt35 = {
    "+Summarized\n+Explanation": {
        "Compressed": 75, "Detailed": 82.5
    },
    "+Ranker\n-Explanation": {
        "Compressed": 82.5, "Detailed": 82.5
    },
    "-Explanation": {
        "Compressed": 85, "Detailed": 85
    }
}

if __name__ == "__main__":
    labels = ["Direct", "Internal-Simple", "Internal-Specified", "External"]
    labels_short = ["Direct", "Simple", "Specified", "External"]
    # colors = ['#580F41', '#A0522D', '#800080', '#8C000F']
    colors = ['#DC143C', '#069AF3', '#9ACD32', '#01153E']
    none = [table_CoT_gpt35[label]["None"] for label in labels]
    compressed = [table_CoT_gpt35[label]["Compressed"] for label in labels]
    detailed = [table_CoT_gpt35[label]["Detailed"] for label in labels]

    # 2. grouped bar chart
    # Also, plots' range would be from 50 to 100
    x = range(len(labels))
    width = 0.3
    fig, ax = plt.subplots()
    plt.axhline(y=62.68, color="#929591", linestyle='--', zorder=0, label="GPT-3.5-baseline")

    ax.bar([i - width for i in x], none, width, label="None", color=colors[0])
    ax.plot([0 + width, 1 + width], [95, 97.5], marker="x", color=colors[3], linestyle='', label="GPT-4")

    ax.bar(x, compressed, width, label="Compressed", color=colors[1])

    ax.bar([i + width for i in x], detailed, width, label="Detailed", color=colors[2])

    ax.set_ylim(50, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("Different CoT methods")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short)
    ax.legend()
    plt.show()

    # Save the plot
    fig.savefig("cot.pdf", format="pdf")
    plt.close()

    ############################################## same plot for rag ##################################################
    labels = [x for x in table_rag_gpt35.keys()]

    x = range(len(labels))
    width = 0.3
    fig, ax = plt.subplots()
    plt.axhline(y=62.68, color="#929591", linestyle='--', zorder=0, label="GPT-3.5-baseline")

    # Extracting data for each label and category
    compressed = [table_rag_gpt35[label]["Compressed"] for label in labels]
    detailed = [table_rag_gpt35[label]["Detailed"] for label in labels]

    ax.bar([i - width/2 for i in x], compressed, width, label="Compressed", color=colors[1])
    ax.plot([2 + width/2], [92.5], marker="x", color=colors[3], linestyle='', label="GPT-4")

    ax.bar([i + width / 2 for i in x], detailed, width, label="Detailed", color=colors[2])
    # ax.plot([0 - width/2,], [72.5, 77.5], marker="o", color="#DC143C", linestyle='', label="Simple-CoT(GPT-3.5)")


    ax.set_ylim(50, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("ICL with Different RAG settings")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.show()

    # Save the plot
    fig.savefig("rag.pdf", format="pdf")
    plt.close()

# if __name__ == "__main__":
#     labels = ["Direct", "Internal-Simple", "Internal-Specified", "External"]
#     labels_short = ["Direct", "Simple", "Specified", "External"]
#     # colors = ['#580F41', '#A0522D', '#800080', '#8C000F']
#     colors = ['#DC143C', '#069AF3', '#9ACD32', '#01153E']
#     none = [table_CoT_gpt35[label]["None"] for label in labels]
#     compressed = [table_CoT_gpt35[label]["Compressed"] for label in labels]
#     detailed = [table_CoT_gpt35[label]["Detailed"] for label in labels]
#
#     # 2. grouped bar chart
#     # Also, plots' range would be from 50 to 100
#     x = range(len(labels))
#     width = 0.3
#     fig, ax = plt.subplots()
#     plt.axhline(y=62.68, color="#929591", linestyle='solid', zorder=0, label="GPT-3.5-baseline")
#     ax.plot([0, 1], [95, 97.5], marker="x", color=colors[3], linestyle='--', label="GPT-4")
#
#     ax.plot(x, none, label="None", marker="o", color=colors[0], linestyle="--")
#     ax.plot(x, compressed, marker="o", linestyle='--', label="Compressed", color=colors[1])
#     ax.plot(x, detailed, marker="o", linestyle='--', label="Detailed", color=colors[2])
#
#     ax.set_ylim(50, 100)
#     ax.set_ylabel("Score (%)")
#     ax.set_title("Different CoT methods")
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels_short)
#     ax.legend()
#     plt.show()
#
#     # Save the plot
#     fig.savefig("cot.pdf", format="pdf")
#     plt.close()
#
#     # same plot for rag
#     labels = [x for x in table_rag_gpt35.keys()]
#
#     x = range(len(labels))
#     width = 0.3
#     fig, ax = plt.subplots()
#     plt.axhline(y=62.68, color="#929591", linestyle='solid', zorder=0, label="GPT-3.5-baseline")
#     ax.plot([2], [92.5], marker="x", linestyle='', color=colors[3], label="GPT-4")
#
#     # Extracting data for each label and category
#     compressed = [table_rag_gpt35[label]["Compressed"] for label in labels]
#     detailed = [table_rag_gpt35[label]["Detailed"] for label in labels]
#
#     ax.plot(x, compressed, marker="o", linestyle='', label="Compressed", color=colors[1])
#
#     ax.plot(x, detailed, marker="o", linestyle='', label="Detailed", color=colors[2])
#
#
#     ax.set_ylim(50, 100)
#     ax.set_ylabel("Score (%)")
#     ax.set_title("RAG scores")
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels)
#     ax.legend()
#     plt.show()
#
#     # Save the plot
#     fig.savefig("rag.pdf", format="pdf")
#     plt.close()



