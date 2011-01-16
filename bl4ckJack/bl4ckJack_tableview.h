 #ifndef TABLEMODEL_H
 #define TABLEMODEL_H

 #include <QAbstractTableModel>
 #include <QPair>
 #include <QList>
 #include <QItemSelection>

 class QSortFilterProxyModel;
 class QItemSelectionModel;

 class TableModel : public QAbstractTableModel
 {
     Q_OBJECT

 public:
     TableModel(QObject *parent=0);
     TableModel(QList< QPair<QString, QString> > listofPairs, QObject *parent=0);

     int rowCount(const QModelIndex &parent) const;
     int columnCount(const QModelIndex &parent) const;
     QVariant data(const QModelIndex &index, int role) const;
     QVariant headerData(int section, Qt::Orientation orientation, int role) const;
     Qt::ItemFlags flags(const QModelIndex &index) const;
     bool setData(const QModelIndex &index, const QVariant &value, int role=Qt::EditRole);
     bool insertRows(int position, int rows, const QModelIndex &index=QModelIndex());
     bool removeRows(int position, int rows, const QModelIndex &index=QModelIndex());
     QList< QPair<QString, QString> > getList();
	 void readFromFile(QString fileName);
     void writeToFile(QString fileName);

 public slots:
     void addEntry(QString name, QString address);
     //void editEntry();
     //void removeEntry();

 signals:
     void selectionChanged (const QItemSelection &selected);

 private:
     QList< QPair<QString, QString> > listOfPairs;
     QSortFilterProxyModel *proxyModel;
 };



 #endif