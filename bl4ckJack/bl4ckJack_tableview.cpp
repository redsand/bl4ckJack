
#include <Qt>
#include <QTableView>
#include "bl4ckJack_tableview.h"
#include <QFile>

TableModel::TableModel(QObject *parent)
     : QAbstractTableModel(parent)
 {
 }

 TableModel::TableModel(QList< QPair<QString, QString> > pairs, QObject *parent)
     : QAbstractTableModel(parent)
 {
     listOfPairs=pairs;


 }

 int TableModel::rowCount(const QModelIndex &parent) const
 {
     Q_UNUSED(parent);
     return listOfPairs.size();
 }

 int TableModel::columnCount(const QModelIndex &parent) const
 {
     Q_UNUSED(parent);
     return 2;
 }
   
 QVariant TableModel::data(const QModelIndex &index, int role) const
 {
     if (!index.isValid())
         return QVariant();

     if (index.row() >= listOfPairs.size() || index.row() < 0)
         return QVariant();

	 if(role == Qt::TextAlignmentRole) {
		if(index.column() == 0)
			return QVariant(Qt::AlignCenter | Qt::AlignVCenter);
		else
			return QVariant(Qt::AlignLeft | Qt::AlignVCenter);
	 }

     if (role == Qt::DisplayRole) {
         QPair<QString, QString> pair = listOfPairs.at(index.row());

         if (index.column() == 0)
             return pair.first;
         else if (index.column() == 1)
             return pair.second;
     }

     return QVariant();
 }

 QVariant TableModel::headerData(int section, Qt::Orientation orientation, int role) const
 {
     if (role != Qt::DisplayRole)
         return QVariant();

     if (orientation == Qt::Horizontal) {
         switch (section) {
             case 0:
                 return tr("Module");

             case 1:
                 return tr("Hash");

             default:
                 return QVariant();
         }
     }
     return QVariant();
 }

 bool TableModel::insertRows(int position, int rows, const QModelIndex &index)
 {
     Q_UNUSED(index);
     beginInsertRows(QModelIndex(), position, position+rows-1);

     for (int row=0; row < rows; row++) {
         QPair<QString, QString> pair(" ", " ");
         listOfPairs.insert(position, pair);
     }

     endInsertRows();
     return true;
 }

 bool TableModel::removeRows(int position, int rows, const QModelIndex &index)
 {
     Q_UNUSED(index);
	 long long row = position+rows-1;
	 if(row < 0)
		 return true;

     beginRemoveRows(QModelIndex(), position, position+rows-1);

     for (int row=0; row < rows; ++row) {
         listOfPairs.removeAt(position);
     }

     endRemoveRows();
     return true;
 }

 bool TableModel::setData(const QModelIndex &index, const QVariant &value, int role)
 {
         if (index.isValid() && role == Qt::EditRole) {
                 int row = index.row();

                 QPair<QString, QString> p = listOfPairs.value(row);

                 if (index.column() == 0)
                         p.first = value.toString();
                 else if (index.column() == 1)
                         p.second = value.toString();
         else
             return false;

         listOfPairs.replace(row, p);
                 emit(dataChanged(index, index));

         return true;
         }

         return false;
 }

 Qt::ItemFlags TableModel::flags(const QModelIndex &index) const
 {
     if (!index.isValid())
         return Qt::ItemIsEnabled;

     return QAbstractTableModel::flags(index) | Qt::ItemIsEditable;
 }

 QList< QPair<QString, QString> > TableModel::getList()
 {
     return listOfPairs;
 }

 void TableModel::addEntry(QString name, QString address)
 {
     QList< QPair<QString, QString> >list = getList();
     QPair<QString, QString> pair(name, address);

     if (!list.contains(pair)) {
         insertRows(0, 1, QModelIndex());
         QModelIndex index = this->index(0, 0, QModelIndex());
         setData(index, name, Qt::EditRole);
         index = this->index(0, 1, QModelIndex());
         setData(index, address, Qt::EditRole);
     } else {
         //qDebug() << "The name " << name << " already exists.";
     }
 }

 /*
 void TableModel::editEntry()
 {
     //QTableView *temp = static_cast<QTableView*>(this);
     //QSortFilterProxyModel *proxy = static_cast<QSortFilterProxyModel*>(this->model());
     QItemSelectionModel *selectionModel = this->selectionModel();

     QModelIndexList indexes = selectionModel->selectedRows();
     QModelIndex index, i;
     QString name;
     QString address;
     int row;

     foreach (index, indexes) {
         row = proxy->mapToSource(index).row();
         i = index(row, 0, QModelIndex());
         QVariant varName = data(i, Qt::DisplayRole);
         name = varName.toString();

         i = index(row, 1, QModelIndex());
         QVariant varAddr = data(i, Qt::DisplayRole);
         address = varAddr.toString();
     }

     AddDialog aDialog;
     aDialog.setWindowTitle(tr("Edit a Contact"));

     aDialog.nameText->setReadOnly(true);
     aDialog.nameText->setText(name);
     aDialog.addressText->setText(address);

     if (aDialog.exec()) {
         QString newAddress = aDialog.addressText->toPlainText();
         if (newAddress != address) {
             i = table->index(row, 1, QModelIndex());
             table->setData(i, newAddress, Qt::EditRole);
         }
     }
 }
*/

 /*
 void TableModel::removeEntry()
 {
     QTableView *temp = static_cast<QTableView*>(currentWidget());
     QSortFilterProxyModel *proxy = static_cast<QSortFilterProxyModel*>(temp->model());
     QItemSelectionModel *selectionModel = temp->selectionModel();

     QModelIndexList indexes = selectionModel->selectedRows();
     QModelIndex index;

     foreach (index, indexes) {
         int row = proxy->mapToSource(index).row();
         this->removeRows(row, 1, QModelIndex());
     }

     if (this->rowCount(QModelIndex()) == 0) {
         //insertTab(0, newAddressTab, "Address Book");
     }
 }
*/
 /*
 void AddressWidget::setupTabs()
 {
     QStringList groups;
     groups << "ABC" << "DEF" << "GHI" << "JKL" << "MNO" << "PQR" << "STU" << "VW" << "XYZ";

     for (int i = 0; i < groups.size(); ++i) {
         QString str = groups.at(i);

         proxyModel = new QSortFilterProxyModel(this);
         proxyModel->setSourceModel(table);
         proxyModel->setDynamicSortFilter(true);

         QTableView *tableView = new QTableView;
         tableView->setModel(proxyModel);
         tableView->setSortingEnabled(true);
         tableView->setSelectionBehavior(QAbstractItemView::SelectRows);
         tableView->horizontalHeader()->setStretchLastSection(true);
         tableView->verticalHeader()->hide();
         tableView->setEditTriggers(QAbstractItemView::NoEditTriggers);
         tableView->setSelectionMode(QAbstractItemView::SingleSelection);

         QString newStr = QString("^[%1].*").arg(str);

         proxyModel->setFilterRegExp(QRegExp(newStr, Qt::CaseInsensitive));
         proxyModel->setFilterKeyColumn(0);
         proxyModel->sort(0, Qt::AscendingOrder);

         connect(tableView->selectionModel(),
             SIGNAL(selectionChanged(const QItemSelection &, const QItemSelection &)),
             this, SIGNAL(selectionChanged(const QItemSelection &)));

         addTab(tableView, str);
     }
 }
*/
 void TableModel::readFromFile(QString fileName)
 {
     QFile file(fileName);

     if (!file.open(QIODevice::ReadOnly)) {
         //QMessageBox::information(this, tr("Unable to open file"),
         //    file.errorString());
         return;
     }

     QList< QPair<QString, QString> > pairs = this->getList();
     QDataStream in(&file);
     in >> pairs;

     if (pairs.isEmpty()) {
         //QMessageBox::information(this, tr("No contacts in file"),
         //    tr("The file you are attempting to open contains no contacts."));
     } else {
         for (int i=0; i<pairs.size(); ++i) {
             QPair<QString, QString> p = pairs.at(i);
             addEntry(p.first, p.second);
         }
     }
 }

 void TableModel::writeToFile(QString fileName)
 {
     QFile file(fileName);

     if (!file.open(QIODevice::WriteOnly)) {
         //QMessageBox::information(this, tr("Unable to open file"), filen.errorString());
         return;
     }

     QList< QPair<QString, QString> > pairs = getList();
     QDataStream out(&file);
     out << pairs;
 }